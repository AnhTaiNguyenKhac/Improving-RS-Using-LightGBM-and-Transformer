import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Tuple
from pathlib import Path


class DataHelper:
    def __init__(self, data_dir: str, raw_dir: str = "raw"):

        self.base = Path(data_dir)  # data diectory
        self.raw_dir = self.base / raw_dir  # raw data directory

    def _load_raw_data(self) -> dict:
        articles = pd.read_csv(self.raw_dir / "articles.csv")
        customers = pd.read_csv(self.raw_dir / "customers.csv")
        inter = pd.read_csv(self.raw_dir / "transactions_train.csv")

        return {"item": articles, "user": customers, "inter": inter}

    # def _encode_id(self, data: dict, map_dir: str) -> dict:
    
    #     if not os.path.isdir(self.base / map_dir):
    #         os.mkdir(self.base / map_dir)

    #     user = data["user"]
    #     item = data["item"]
    #     inter = data["inter"]

    #     user_id2index_path = self.base / map_dir / "user_id2index.pkl"
    #     user_index2id_path = self.base / map_dir / "user_index2id.pkl"
    #     item_id2index_path = self.base / map_dir / "item_id2index.pkl"
    #     item_index2id_path = self.base / map_dir / "item_index2id.pkl"

    #     if not os.path.exists(user_id2index_path):
    #         user_id2index_dict = dict(zip(user["customer_id"], user.index + 1))
    #         pickle.dump(user_id2index_dict, open(user_id2index_path, "wb"))
    #     else:
    #         user_id2index_dict = pickle.load(open(user_id2index_path, "rb"))

    #     if not os.path.exists(user_index2id_path):
    #         user_index2id_dict = dict(zip(user.index + 1, user["customer_id"]))
    #         pickle.dump(user_index2id_dict, open(user_index2id_path, "wb"))
    #     else:
    #         user_index2id_dict = pickle.load(open(user_index2id_path, "rb"))

    #     if not os.path.exists(item_id2index_path):
    #         item_id2index_dict = dict(zip(item["article_id"], item.index + 1))
    #         pickle.dump(item_id2index_dict, open(item_id2index_path, "wb"))
    #     else:
    #         item_id2index_dict = pickle.load(open(item_id2index_path, "rb"))

    #     if not os.path.exists(item_index2id_path):
    #         item_index2id_dict = dict(zip(item.index + 1, item["article_id"]))
    #         pickle.dump(item_index2id_dict, open(item_index2id_path, "wb"))
    #     else:
    #         item_index2id_dict = pickle.load(open(item_index2id_path, "rb"))

    #     inter["customer_id"] = inter["customer_id"].map(user_id2index_dict)
    #     inter["article_id"] = inter["article_id"].map(item_id2index_dict)
    #     user["customer_id"] = user["customer_id"].map(user_id2index_dict)
    #     item["article_id"] = item["article_id"].map(item_id2index_dict)

    #     data["user"] = user
    #     data["item"] = item
    #     data["inter"] = inter

    #     return data
    def _encode_id(self, data: dict, map_dir: str) -> dict:
        if not os.path.isdir(self.base / map_dir):
            os.mkdir(self.base / map_dir)

        user = data["user"]
        item = data["item"]
        inter = data["inter"]

        # Đảm bảo ID là string
        user["customer_id"] = user["customer_id"].astype(str)
        item["article_id"] = item["article_id"].astype(str)
        inter["customer_id"] = inter["customer_id"].astype(str)
        inter["article_id"] = inter["article_id"].astype(str)

        # Lấy ID duy nhất
        unique_user_ids = user["customer_id"].drop_duplicates().reset_index(drop=True)
        unique_item_ids = item["article_id"].drop_duplicates().reset_index(drop=True)

        # Tạo dict: ID → index
        user_id2index_dict = pd.Series(unique_user_ids.index + 1, index=unique_user_ids).to_dict()
        item_id2index_dict = pd.Series(unique_item_ids.index + 1, index=unique_item_ids).to_dict()

        # Ngược lại: index → ID
        user_index2id_dict = pd.Series(unique_user_ids.values, index=unique_user_ids.index + 1).to_dict()
        item_index2id_dict = pd.Series(unique_item_ids.values, index=unique_item_ids.index + 1).to_dict()

        # Ghi ra file
        pickle.dump(user_id2index_dict, open(self.base / map_dir / "user_id2index.pkl", "wb"))
        pickle.dump(user_index2id_dict, open(self.base / map_dir / "user_index2id.pkl", "wb"))
        pickle.dump(item_id2index_dict, open(self.base / map_dir / "item_id2index.pkl", "wb"))
        pickle.dump(item_index2id_dict, open(self.base / map_dir / "item_index2id.pkl", "wb"))

        # Áp dụng map
        inter["customer_id"] = inter["customer_id"].map(user_id2index_dict)
        inter["article_id"] = inter["article_id"].map(item_id2index_dict)
        user["customer_id"] = user["customer_id"].map(user_id2index_dict)
        item["article_id"] = item["article_id"].map(item_id2index_dict)

        data["user"] = user
        data["item"] = item
        data["inter"] = inter

        return data

    def _base_features(self, data: dict) -> dict:
       
        item = data["item"]
        user = data["user"]
        trans = data["inter"]

        def set_gender_flg(x):
            female_pro_types = [
                "Bra",
                "Underwear Tights",
                "Leggings/Tights",
                "Hair clip",
                "Hair string",
                "Hair/alice band",
                "Bikini top",
                "Skirt",
                "Dress",
                "Earring",
                "Alice band",
                "Straw hat",
                "Necklace",
                "Ballerinas",
                "Blouse",
                "Beanie",
                "Giftbox",
                "Pumps",
                "Bootie",
                "Heeled sandals",
                "Nipple covers",
                "Hair ties",
                "Underwear corset",
                "Bra extender",
                "Underdress",
                "Underwear set",
                "Sarong",
                "Leg warmers",
                "Hairband",
                "Tote bag",
                "Earrings",
                "Flat shoes",
                "Heels",
                "Cap",
                "Shoulder bag",
                "Headband",
                "Baby Bib",
                "Cross-body bag",
                "Bumbag",
            ]
            # x["article_gender"] = 0  # * 0 for not divided, 1 for male, 2 for female
            # if x["index_group_name"] == "Ladieswear":
            #     x["article_gender"] = 2
            # elif x["index_group_name"] == "Menswear":
            #     x["article_gender"] = 1
            # else:
            #     if (
            #         "boy" in x["department_name"].lower()
            #         or "men" in x["department_name"].lower()
            #     ):
            #         x["article_gender"] = 1
            #     if (
            #         "girl" in x["department_name"].lower()
            #         or "ladies" in x["department_name"].lower()
            #         or x["product_type_name"] in female_pro_types
            #     ):
            #         x["article_gender"] = 2
            # return x
            x["article_gender"] = 0  # 0: not divided, 1: male, 2: female

            index_group = str(x.get("index_group_name", "")).lower()
            department = str(x.get("department_name", "")).lower()
            product_type = str(x.get("product_type_name", "")).strip()

            if index_group == "ladieswear":
                x["article_gender"] = 2
            elif index_group == "menswear":
                x["article_gender"] = 1
            else:
                if "boy" in department or "men" in department:
                    x["article_gender"] = 1
                if "girl" in department or "ladies" in department or product_type in female_pro_types:
                    x["article_gender"] = 2
            return x

        # * Recognize article gender
        item = item.apply(set_gender_flg, axis=1)

        # * Seasonal Articles
        summer = [
            "Sunglasses",
            "Hat/brim",
            "Sandals",
            "Flat shoe",
            "Heeled sandals",
            "Polo shirt",
            "Dress",
            "T-shirt",
            "Skirt",
            "Vest top",
            "Swimwear top",
            "Swimsuit",
            "Swimwear bottom",
            "Bikini top",
            "Shorts",
        ]
        winter = [
            "Beanie",
            "Felt hat",
            "Outdoor overall",
            "Long John",
            "Pyjama bottom",
            "Hat/beanie",
            "Leggings/Tights",
            "Hoodie",
            "Underwear Tights",
            "Pyjama set",
            "Boots",
            "Cardigan",
            "Sweater",
            "Jacket",
            "Scarf",
            "Coat",
            "Gloves",
            "Outdoor Waistcoat",
        ]
        item["season_type"] = 0
        item.loc[item["product_type_name"].isin(summer), "season_type"] = 1
        item.loc[item["product_type_name"].isin(winter), "season_type"] = 2

        # * Recognize user gender
        trans = pd.merge(
            trans,
            item[["article_id", "article_gender", "product_type_name"]],
            on="article_id",
            how="left",
        )
        ttl_cnt = trans.groupby(["customer_id"]).size().reset_index(name="ttl_cnt")
        gender_sale = (
            trans.groupby(["customer_id", "article_gender"])
            .size()
            .reset_index(name="cnt")
        )
        gender_sale = gender_sale.merge(ttl_cnt, on=["customer_id"], how="left")
        gender_sale["ratio"] = gender_sale["cnt"] / gender_sale["ttl_cnt"]
        gender_sale = pd.pivot_table(
            gender_sale, values="ratio", index="customer_id", columns=["article_gender"]
        )
        gender_sale = gender_sale.reset_index()
        gender_sale["user_gender"] = 0
        gender_sale.loc[gender_sale[1] >= 0.8, "user_gender"] = 1  # * male
        gender_sale.loc[gender_sale[2] >= 0.8, "user_gender"] = 2  # * female
        user = user.merge(
            gender_sale[["customer_id", "user_gender"]], on="customer_id", how="left"
        )
        user["user_gender"] = user["user_gender"].fillna(0)

        data["item"] = item
        data["user"] = user
        return data

    def _transform_feats(self, data: dict) -> dict:
       
        inter = data["inter"]
        user = data["user"]
        item = data["item"]
        user["age"] = user["age"].fillna(0)
        user = user.fillna(-1)

        # Transactions
        inter["price"] = inter["price"].astype("float32")
        inter["sales_channel_id"] = inter["sales_channel_id"].astype("int8")

        # Customers
        user_sparse_feats = [x for x in user.columns if x not in ["age", "user_gender"]]
        for feat in tqdm(
            [x for x in user_sparse_feats if x != "customer_id"],
            "Encode User Sparse Feats",
        ):
            lbe = LabelEncoder()
            user[feat] = lbe.fit_transform(user[feat].astype(str)) + 1
            user[feat] = user[feat].astype("int32")
        user["age"] = user["age"].astype("int8")

        # Articles
        item_sparse_feats = [
            "article_id",
            "product_code",
            "product_type_no",
            "product_group_name",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
            "article_gender",
            "season_type",
        ]
        for feat in tqdm(
            [
                x
                for x in item_sparse_feats
                if x not in ["article_id", "article_gender", "season_type"]
            ],
            "Encode Item Sparse Feats",
        ):
            lbe = LabelEncoder()
            item[feat] = lbe.fit_transform(item[feat].astype(str)) + 1
            item[feat] = item[feat].astype("int32")

        data["inter"] = inter
        data["user"] = user
        data["item"] = item[item_sparse_feats]

        return data

    def save_data(self, data: dict, name: str):
       
        path = self.base / "processed" / name
        if not os.path.exists(path):
            os.mkdir(path)
        data["user"].to_parquet(path / "user.pqt")
        data["item"].to_parquet(path / "item.pqt")
        data["inter"].to_parquet(path / "inter.pqt")

    def load_data(self, name: str) -> dict:
       
        path = self.base / "processed" / name
        if not os.path.exists(path):
            raise OSError(f"{path} does not exist.")
        data = {}
        data["user"] = pd.read_parquet(path / "user.pqt")
        data["item"] = pd.read_parquet(path / "item.pqt")
        data["inter"] = pd.read_parquet(path / "inter.pqt")

        return data

    def preprocess_data(self, save: bool = True, name: str = "encoded_full") -> dict:
      
        data = self._load_raw_data()
        data = self._encode_id(data, "index_id_map")
        data = self._base_features(data)
        data = self._transform_feats(data)
        if save:
            self.save_data(data, name)
        return data

    def split_data(
        self,
        trans_data: pd.DataFrame,
        train_end_date: str,
        valid_end_date: str,
        item_id: str = "article_id",
    ) -> Tuple[pd.DataFrame]:

        if item_id not in trans_data.columns:
            raise KeyError(f"{item_id} is not one of the columns")

        train_set = trans_data.loc[trans_data["t_dat"] < train_end_date]
        valid_set = trans_data.loc[
            (train_end_date <= trans_data["t_dat"])
            & (trans_data["t_dat"] < valid_end_date)
        ]
        valid_set = (
            valid_set.groupby(["customer_id"])[item_id].apply(list).reset_index()
        )

        return train_set, valid_set
