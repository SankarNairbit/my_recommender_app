import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import logging
import hopsworks


class Transformer(object):

    def __init__(self):
        # Connect to the Hopsworks
        project = hopsworks.login()
        ms = project.get_model_serving()
        mr = project.get_model_registry()

        # Retrieve the 'customers' feature view
        self.fs = project.get_feature_store()
        self.customer_fv = self.fs.get_feature_view(
            name="customers", 
            version=1,
        )
        self.customer_fv.init_serving(1)

        # Retrieve 'transactions' feature group.
        self.transactions_fg = self.fs.get_feature_group(
            name="transactions", 
            version=1,
        )

        # Retrieve the 'articles' feature view
        self.articles_fv = self.fs.get_feature_view(
            name="articles",
            version=1,
        )  

        # Get list of feature names for articles
        self.articles_features = [feat.name for feat in self.articles_fv.schema]

        # Retrieve the 'candidate_embeddings' feature view
        self.candidate_index = self.fs.get_feature_view(
            name="candidate_embeddings",
            version=1,
        )

        model = mr.get_model(
            name="ranking_model",
            version=1,
        )

        # Download the saved model files to a local directory
        saved_model_dir = model.download()

        self.model = joblib.load(saved_model_dir + "/ranking_model.pkl")

        self.ranking_fv = model.get_feature_view(init=False)
        self.ranking_fv.init_batch_scoring(1)

        # Get the names of features expected by the ranking model
        self.ranking_model_feature_names = [
            feature.name 
            for feature 
            in self.ranking_fv.schema 
            if feature.name != 'label'
        ]


    def preprocess(self, inputs):
        # Check if the input data contains a key named "instances"
        # and extract the actual data if present
        inputs = inputs["instances"] if "instances" in inputs else inputs
        inputs = inputs[0]

        # Extract customer_id and transaction_date from the inputs
        customer_id = inputs["customer_id"]
        transaction_date = inputs["transaction_date"]

        # Extract month from the transaction_date
        month_of_purchase = datetime.fromisoformat(inputs.pop("transaction_date"))

        # Get customer features
        customer_features = self.customer_fv.get_feature_vector(
            {"customer_id": customer_id},
            return_type="pandas",
        )

        # Enrich inputs with customer age
        inputs["customers_age"] = customer_features.age.values[0]  

        # Calculate the sine and cosine of the month_of_purchase
        month_of_purchase = datetime.strptime(
            transaction_date, "%Y-%m-%dT%H:%M:%S.%f"
        ).month

        # Calculate the sine and cosine components for the month_of_purchase using on-demand transformation present in "ranking" feature view.
        feature_vector = self.ranking_fv._batch_scoring_server.compute_on_demand_features(
            feature_vectors=pd.DataFrame([inputs]), request_parameters={"trans_month": month_of_purchase}
        ).to_dict(orient="records")[0]

        inputs["month_sin"] = feature_vector["trans_month_sin"]
        inputs["month_cos"] = feature_vector["trans_month_cos"]

        return {"instances": [inputs]}


    def postprocess(self, query_outputs):

        inputs = query_outputs[0]

        # Extract customer_id from inputs
        customer_id = inputs["customer_id"]

        # Search for candidate items
        neighbors = self.candidate_index.find_neighbors(
            inputs["query_emb"],
            k=100,
        )
        neighbors = [neighbor[0] for neighbor in neighbors]

        # Get IDs of items already bought by the customer
        already_bought_items_ids = (
            self.transactions_fg.select("article_id").filter(
                self.transactions_fg.customer_id==customer_id
            ).read(dataframe_type="pandas").values.reshape(-1).tolist()
        )

        # Filter candidate items to exclude those already bought by the customer
        item_id_list = [
            str(item_id)
            for item_id in neighbors
            if str(item_id) not in already_bought_items_ids
        ]
        item_id_df = pd.DataFrame({"article_id": item_id_list})

        # Retrieve Article data for candidate items
        articles_data = [
            self.articles_fv.get_feature_vector({"article_id": item_id})
            for item_id in item_id_list
        ]

        logging.info("✅ Articles Data Retrieved!")

        articles_df = pd.DataFrame(
            data=articles_data,
            columns=self.articles_features,
        )

        # Join candidate items with their features
        ranking_model_inputs = item_id_df.merge(
            articles_df,
            on="article_id",
            how="inner",
        )

        logging.info("✅ Inputs are almost ready!")

        # Add customer features
        customer_features = self.customer_fv.get_feature_vector(
                {"customer_id": customer_id},
                return_type="pandas",
            )

        ranking_model_inputs["age"] = customer_features.age.values[0]
        ranking_model_inputs["trans_month_sin"] = inputs["month_sin"]
        ranking_model_inputs["trans_month_cos"] = inputs["month_cos"]

        # Select only the features required by the ranking model
        ranking_model_inputs = ranking_model_inputs[self.ranking_model_feature_names]

        logging.info("✅ Inputs are ready!")

        features = ranking_model_inputs.values.tolist()
        article_ids = item_id_list

        # Log the extracted features
        logging.info("predict -> " + str(features))

        # Log the extracted article ids
        logging.info(f'Article IDs: {article_ids}')

        logging.info(f"🚀 Predicting...")

        # Predict probabilities for the positive class
        scores = self.model.predict_proba(features).tolist()

        # Get scores of positive class
        scores = np.asarray(scores)[:,1].tolist() 

        logging.info("✅ Predictions are ready!")

        # Merge prediction scores and corresponding article IDs into a list of tuples
        ranking = list(zip(scores, article_ids))

        # Sort the ranking list by score in descending order
        ranking.sort(reverse=True)

        # Return the sorted ranking list
        return {
            "ranking": ranking,
        }
