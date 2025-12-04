import pandas as pd

df = pd.read_csv('autism_cleaned_updated.csv')


def drop_unneeded_columns(df : pd.DataFrame) -> pd.DataFrame:
    """
    Drops unneeded columns from the DataFrame.
    Parameters:
    df (DataFrame): a pandas DataFrame
    Returns:
    DataFrame: a pandas DataFrame with unneeded columns dropped
    """
    return df.drop(columns=['Title','Link to Publication', 'Case Identification Method', 'Case Criterion',
                            'Confidence Interval (CI)',	'Male:Female Sex Ratio', 'Non-Hispanic White:Hispanic Prevalence Ratio',	
                            'White:Black Prevalence Ratio',	'Diagnosis Age Range (months)',	'Diagnosis Mean Age (months)',	
                            'Diagnosis Median Age (months)',	'IQ Score <70 (%)',	'Adaptive Score <70 (%)',	
                            'Non-Verbal or Minimally Verbal (%)',	'Percentage of Individual Co-occurring Conditions',	
                            'Autism Types Included',	'Link to Publication',	'CDC Calculated Values',
                            'Sample Size', 'Number of Cases', 'ASD Prevalence Estimate per 1,000', 'Age Range'
])


def fill_areas(df : pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in 'Area(s)' column with 'Unknown'.
    Parameters:
    df (DataFrame): a pandas DataFrame
    Returns:
    DataFrame: a pandas DataFrame with 'Area(s)' column filled with 'Unknown'
    """
    result = df.copy().rename(columns={"area_s": "Area(s)"})
    result['Area(s)'] = result['Area(s)'].fillna('Unknown')
    return result


def fill_missing_study_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing Year Started / Year Ended with Year Published
    Parameters:
    df (DataFrame): a pandas DataFrame
    Returns:
    DataFrame: a pandas DataFrame with Year Started / Year Ended filled with Year Published
    """
    result = df.copy().rename(columns={'year_published': 'Year Published'})
    result['Year Started'] = result['Year Started'].fillna(result['Year Published'])
    result['Year Ended']   = result['Year Ended'].fillna(result['Year Published'])
    return result