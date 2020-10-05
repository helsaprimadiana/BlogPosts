import vaex
import glob
import pyarrow.parquet as pq
from vaex.dataframe import DataFrame
from datetime import datetime
import time
from functools import wraps
from time import time


def get_checkouts(file_type: str) -> DataFrame:
    if file_type == "multi-csv":
        return vaex.open(
            "./data/checkouts/Checkouts_By_Title_Data_Lens_*.csv",
            header=0,
            dtype={
                "BibNum": "int64",
                "Title": "str",
                "Author": "str",
                "Subjects": "str",
                "ItemBarcode": "str",
                "ItemType": "str",
            },
        )
    if file_type == "single_csv":
        return vaex.from_csv(
            "./data/checkouts/Checkouts.csv",
            header=0,
            dtype={
                "BibNum": "int64",
                "Title": "str",
                "Author": "str",
                "Subjects": "str",
                "ItemBarcode": "str",
                "ItemType": "str",
            },
        )
    if file_type == "parquet":
        return vaex.open("./data/checkouts/checkouts_parquet/*.parquet")
    else:
        return vaex.open("./data/checkouts/checkouts.hdf5")


def get_inventory_no_dedup(file_type: str) -> DataFrame:
    if "csv" in file_type:
        return vaex.from_csv(
            "./data/checkouts/Library_Collection_Inventory.csv", header=0
        )
    if file_type == "parquet":
        return vaex.open("./data/checkouts/inventory_parquet/*.parquet")
    else:
        return vaex.open("./data/checkouts/inventory.hdf5")


def get_inventory(file_type: str) -> DataFrame:
    if "csv" in file_type:
        df = vaex.open("./data/checkouts/Library_Collection_Inventory.csv")
        df = df.to_pandas_df()
        df = df.drop_duplicates("BibNum")
        return vaex.from_pandas(df)
    if file_type == "parquet":
        df = vaex.open("./data/checkouts/inventory_parquet/*.parquet")
        df = df.to_pandas_df()
        df = df.drop_duplicates("BibNum")
        return vaex.from_pandas(df)
    else:
        df = vaex.open("./data/checkouts/inventory.hdf5")
        df = df.to_pandas_df()
        df = df.drop_duplicates("BibNum")
        return vaex.from_pandas(df)


def writter_csv(df: DataFrame) -> None:
    df.export_csv("./output/output.csv")


def writter_parquet(df: DataFrame) -> None:
    df.export_parquet("./output/parquet_output")


def writter_hdf5(df: DataFrame) -> None:
    df.export_hdf5("./output/hdf5_file.hdf5")


def to_dt(x):
    return datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p")


def add_formatted_checkout(file_type: str) -> DataFrame:
    df = get_checkouts(file_type)
    df["CheckoutDateTime_formatted"] = df.apply(to_dt, arguments=[df.CheckoutDateTime])
    df["CheckoutDateTime_formatted"] = df.CheckoutDateTime_formatted.astype(
        "datetime64[ns]"
    )
    return df


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        with open("./results_vaex.txt", "a+") as fl:
            ts = time()
            result = f(*args, **kw)
            te = time()
            fl.write("{0}|args:{1}|{3}\n".format(f.__name__, args, kw, te - ts))
            return result

    return wrap


@timing
def lazy_read(file_type):
    get_checkouts(file_type)


@timing
def lazy_read_head(file_type):
    get_checkouts(file_type).head()


@timing
def read_write_inventory(file_type, writter):
    writter(get_inventory_no_dedup(file_type))


@timing
def read_dedup_write_inventory(file_type, writter):
    writter(get_inventory(file_type))


@timing
def checkouts_get_count(file_type):
    get_checkouts(file_type).count()


@timing
def checkouts_format_dt_first_10(file_type):
    add_formatted_checkout(file_type).sort(
        "CheckoutDateTime_formatted", ascending=True
    ).head(10)


@timing
def read_write_checkouts(file_type, writter):
    writter(get_checkouts(file_type))


@timing
def read_fix_date_write_checkouts(file_type, writter):
    writter(add_formatted_checkout(file_type))


@timing
def checkouts_fix_date_min(file_type):
    df = add_formatted_checkout(file_type)

    df = df.min("CheckoutDateTime_formatted")
    print(df)


@timing
def read_both_join_head(file_type):
    checkouts = get_checkouts(file_type)[["BibNumber", "CheckoutDateTime"]]
    inventory = get_inventory(file_type)[["BibNum", "Author"]]
    merged = checkouts.join(
        inventory,
        left_on="BibNumber",
        right_on="BibNum",
        how="inner",
        allow_duplication=True,
    )
    merged.head()


@timing
def read_both_anti_join_head(file_type):
    checkouts = get_checkouts(file_type)[["BibNumber", "CheckoutDateTime"]]
    inventory = get_inventory(file_type)[["BibNum", "Author"]]
    merged = checkouts.join(
        inventory,
        left_on="BibNumber",
        right_on="BibNum",
        how="right",
        allow_duplication=True,
    )
    merged[merged.BibNumber.isna()].head()


@timing
def read_inv_explode_write(file_type, writter):
    inventory = get_inventory(file_type)[["BibNum", "Subjects"]]
    inventory = inventory.to_pandas_df()
    inventory["Subject"] = inventory["Subjects"].str.split(",").fillna("")
    inventory = inventory.explode("Subject").reset_index()
    inventory = vaex.from_pandas(inventory)
    writter(inventory)


@timing
def read_inv_explode_write(file_type, writter):
    inventory = get_inventory(file_type)[["BibNum", "Subjects"]]
    inventory = inventory.to_pandas_df()
    inventory["Subject"] = inventory["Subjects"].str.split(",").fillna("")
    inventory = inventory.explode("Subject").reset_index()
    inventory = vaex.from_pandas(inventory)
    writter(inventory)


@timing
def popular_subject(file_type):
    inventory = get_inventory(file_type)[["BibNum", "Subjects"]]
    inventory = inventory.to_pandas_df()
    inventory["Subject"] = inventory["Subjects"].str.split(",").fillna("")
    inventory = inventory.explode("Subject").reset_index()
    inventory = vaex.from_pandas(inventory)
    inventory = (
        inventory.groupby(by="Subject")
        .agg({"BibNum": "nunique"})
        .sort("BibNum", ascending=False)
    )
    inventory.head(10)


@timing
def popular_subject_from_taken_books(file_type):
    checkouts = get_checkouts(file_type)[["BibNumber", "CheckoutDateTime"]]
    inventory = get_inventory(file_type)[["BibNum", "Subjects"]]
    inventory = inventory.to_pandas_df()
    inventory["Subject"] = inventory["Subjects"].str.split(",").fillna("")
    inventory = inventory.explode("Subject")
    inventory = vaex.from_pandas(inventory)
    merged = checkouts.join(
        inventory,
        left_on="BibNumber",
        right_on="BibNum",
        how="inner",
        allow_duplication=True,
    )
    merged = merged.groupby(by="Subject").agg({"BibNum": "nunique"})
    merged.head()


def main():
    file_types = ["hdf5", "parquet", "single-csv"]  # ,"multi-csv"]
    wr = [writter_hdf5, writter_csv, writter_parquet]
    for file_type in file_types:
        lazy_read(file_type)
        lazy_read_head(file_type)
        checkouts_get_count(file_type)
        checkouts_format_dt_first_10(file_type)
        checkouts_fix_date_min(file_type)
        read_both_join_head(file_type)
        read_both_anti_join_head(file_type)
        popular_subject(file_type)
        popular_subject_from_taken_books(file_type)
        for writter in wr:
            read_write_inventory(file_type, writter)
            read_dedup_write_inventory(file_type, writter)
            read_write_checkouts(file_type, writter)
            read_fix_date_write_checkouts(file_type, writter)
            read_inv_explode_write(file_type, writter)


if __name__ == "__main__":
    f = open("./results_vaex.txt", "w")
    f.close()
    main()
