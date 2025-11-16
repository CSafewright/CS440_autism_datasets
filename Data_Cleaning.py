!pip install sodapy

from sodapy import Socrata
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import regexp_replace, col, element_at, regexp_extract, when, expr, mean
import pandas as pd

spark = SparkSession.builder.getOrCreate()

def get_data_cdc_api():
    client = Socrata('data.cdc.gov', None)
    return client.get("9mw4-6adp", limit=2000)

def drop_unneeded_columns(df : DataFrame) -> DataFrame:
    return df.drop('title','link_to_publication', 'cdc_calculated_values',
         'adaptive_score_70', 'diagnosis_age_range_months',
         'diagnosis_median_age_months', 'diagnosis_mean_age_months',
         'non_verbal_or_minimally_verbal', 'percentage_of_individual',
         'iq_score_70', 'non_hispanic_white_hispanic',
         'white_black_prevalence_ratio', 'autism_types_included',
         'case_criterion', 'confidence_interval_ci', 'case_identification_method')
    
def fill_areas(df : DataFrame) -> DataFrame:
    return df.withColumnRenamed("area_s", "Area(s)").fillna({'Area(s)': 'Unknown'})

def clean_age(df : DataFrame) -> DataFrame:
    return df.filter(col('Age Range') != "18 to 64")

def clean_sample_size(df : DataFrame) -> DataFrame:
    return (
        df.dropna(subset=["sample_size"])
        .withColumn("sample_size", regexp_replace(col("sample_size"), ",", ""))
        .withColumnRenamed('sample_size', 'Sample Size')
    )

def clean_number_cases(df : DataFrame) -> DataFrame:
    return (
        df.dropna(subset=["number_of_cases"])
        .withColumn("number_of_cases", regexp_replace(col("number_of_cases"), ",", ""))
        .withColumnRenamed('number_of_cases', 'Number of Cases')
    )

def split_study_years(df: DataFrame) -> DataFrame:
    """
    Extracts first and last 4-digit years from 'Study Years' into 'Year Started' and 'Year Ended'.
    """
    return (
        df.withColumnRenamed('study_years', 'Study Years')
        .withColumn("Year Started", regexp_extract(col("Study Years"), r"(\d{4})", 1))
        .withColumn("Year Ended", regexp_extract(col("Study Years"), r"(\d{4})$", 1))
    )

def fill_missing_study_years(df : DataFrame) -> DataFrame:
    # Fill missing Year Started / Year Ended with Year Published
    return (
        df.withColumnRenamed('year_published', 'Year Published')
        .withColumn("Year Started", when(col("Year Started").isNull(), col("Year Published")).otherwise(col("Year Started")))
        .withColumn("Year Ended", when(col("Year Ended").isNull(), col("Year Published")).otherwise(col("Year Ended")))
    )


def parse_age_range(df: DataFrame) -> DataFrame:
    """
    Extracts Youngest and Oldest Age from 'Age Range' column and casts safely to double.
    """
    pattern = r'(\d+(?:\s*\.\s*\d+)?)\s*to\s*(\d+(?:\s*\.\s*\d+)?)'

    return (
        df
        .withColumnRenamed('age_range', 'Age Range')
        # Extract Youngest and Oldest Age
        .withColumn("Youngest Age", regexp_extract(col("Age Range"), pattern, 1))
        .withColumn("Oldest Age",   regexp_extract(col("Age Range"), pattern, 2))
        # Remove spaces like '5 .5'
        .withColumn("Youngest Age", regexp_replace(col("Youngest Age"), r"\s+", ""))
        .withColumn("Oldest Age",   regexp_replace(col("Oldest Age"), r"\s+", ""))
        # Replace empty strings with NULL
        .withColumn("Youngest Age", when(col("Youngest Age") == "", None).otherwise(col("Youngest Age")))
        .withColumn("Oldest Age",   when(col("Oldest Age") == "", None).otherwise(col("Oldest Age")))
        # Cast safely to double using try_cast
        .withColumn("Youngest Age", expr("try_cast(`Youngest Age` as double)"))
        .withColumn("Oldest Age",   expr("try_cast(`Oldest Age` as double)"))
        # Fill missing Oldest Age with Youngest Age
        .withColumn("Oldest Age", when(col("Oldest Age").isNull(), col("Youngest Age")).otherwise(col("Oldest Age")))
    )


def age_cast_numeric(df: DataFrame) -> DataFrame:
    """
    Safely casts columns to numeric using try_cast
    """
    cols = ['Sample Size', 'Number of Cases']
    for c in cols:
        df = df.withColumn(c, expr(f"try_cast(`{c}` as {'double'})"))
    return df

def drop_age_range(df : DataFrame) -> DataFrame:
    return df.drop("Age Range")

def drop_study_years(df : DataFrame) -> DataFrame:
    return df.drop("Study Years")

def convert_to_nums(df : DataFrame) -> DataFrame:
    int_cols = ['Year Started', 'Year Ended', 'Sample Size', 'Number of Cases']
    double_cols = ['Youngest Age', 'Oldest Age']
    for c in int_cols:
        df = df.withColumn(c, col(c).cast('int'))
    for c in double_cols:
        df = df.withColumn(c, col(c).cast('double'))
    return df

def fill_age_means(df: DataFrame) -> DataFrame:
    """
    Computes the mean of Youngest Age and Oldest Age columns,
    rounds to 1 decimal place, and fills missing values with the means.
    """
    avg_vals = df.agg(
        mean("Youngest Age").alias("avg_youngest"),
        mean("Oldest Age").alias("avg_oldest")
    ).collect()[0]
    avg_youngest = round(avg_vals["avg_youngest"], 1) if avg_vals["avg_youngest"] is not None else None
    avg_oldest   = round(avg_vals["avg_oldest"], 1) if avg_vals["avg_oldest"] is not None else None
    return df.fillna({
        "Youngest Age": avg_youngest,
        "Oldest Age":   avg_oldest
    })

def fill_mf_ratio_mean(df: DataFrame) -> DataFrame:
    """
    Computes the mean of 'Male:Female Sex Ratio' column and fills missing values with it.
    """
    avg_val = df.agg(mean("male_female_sex_ratio").alias("avg_ratio")).collect()[0]["avg_ratio"]
    return df.fillna({"male_female_sex_ratio": avg_val}).withColumnRenamed('male_female_sex_ratio', 'Male:Female Sex Ratio')

spark = SparkSession.builder.getOrCreate()

#data = pd.read_csv("https://raw.githubusercontent.com/CSafewright/CS440_autism_datasets/refs/heads/main/autism_prevalence_studies_20250928.csv")
data = get_data_cdc_api()

df = spark.createDataFrame(data)
df = drop_unneeded_columns(df)
#df = clean_age(df)
df = fill_areas(df)
df = clean_sample_size(df)
df = clean_number_cases(df)
df = split_study_years(df)
df = fill_missing_study_years(df)
df = parse_age_range(df)
df = age_cast_numeric(df)
df = drop_age_range(df)
df = drop_study_years(df)
df = convert_to_nums(df)
df = fill_age_means(df)
df = fill_mf_ratio_mean(df)
df = df.withColumnRenamed('asd_prevalence_estimate_per', 'ASD Prevalence Estimate per 1,000')
df = df.withColumnRenamed('author', 'Author')
df = df.withColumnRenamed('country', 'Country')
print(df.dtypes)
df.toPandas().to_csv('autism_cleaned_updated.csv', index=False)


display(df)