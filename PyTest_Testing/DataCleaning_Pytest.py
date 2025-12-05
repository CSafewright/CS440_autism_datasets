import DataCleaning_test as dct
import pytest
import pandas as pd


data = pd.read_csv('autism_prevalence_studies_20250928.csv')
df = pd.DataFrame(data)
print(df.columns.tolist())

@pytest.mark.parametrize('input, expected', [(df, df[[
                                                     'Author',
                                                     'Year Published',
                                                     'Country', 
                                                     'Area(s)', 
                                                     'Study Years',
                                             ]])
                                                  ])
def test_drop_unneeded_columns(input, expected):
    assert dct.drop_unneeded_columns(input).equals(expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        # 1 missing
        (
            pd.DataFrame({
                'Author': ['A', 'B', 'C'],
                'Area(s)': ['USA', None, 'Europe']
            }),
            pd.DataFrame({
                'Author': ['A', 'B', 'C'],
                'Area(s)': ['USA', 'Unknown', 'Europe']
            })
        ),
        # 0 missing
        (
            pd.DataFrame({
                'Author': ['X', 'Y'],
                'Area(s)': ['Asia', 'Africa']
            }),
            pd.DataFrame({
                'Author': ['X', 'Y'],
                'Area(s)': ['Asia', 'Africa']
            })
        ),
        # All missing
        (
            pd.DataFrame({
                'Author': ['M', 'N'],
                'Area(s)': [None, None]
            }),
            pd.DataFrame({
                'Author': ['M', 'N'],
                'Area(s)': ['Unknown', 'Unknown']
            })
        ),
    ]
)
def test_fill_areas(input, expected):
    result = dct.fill_areas(input)
    assert result.equals(expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        # 0 missing
        (
            pd.DataFrame({
                'Title': ['StudyA'],
                'Year Started': [1995],
                'Year Ended': [1997],
                'year_published': [1995]
            }),
            pd.DataFrame({
                'Title': ['StudyA'],
                'Year Started': [1995],
                'Year Ended': [1997],
                'Year Published': [1995]
            })
        ),
        # All missing
        (
            pd.DataFrame({
                'Title': ['StudyX'],
                'Year Started': [None],
                'Year Ended': [None],
                'year_published': [2001]
            }),
            pd.DataFrame({
                'Title': ['StudyX'],
                'Year Started': [2001],
                'Year Ended': [2001],
                'Year Published': [2001]
            })
        ),
    ]
)
def test_fill_missing_study_years(input, expected):
    result = dct.fill_missing_study_years(input)
    assert result.equals(expected)