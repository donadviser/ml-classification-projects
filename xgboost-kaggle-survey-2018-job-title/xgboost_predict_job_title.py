# Using XGBOOST to predict whether a respondent to a Kaggle survey 20218
# job tittle is "Data Scientist" or "Software Engineer"

# import the libraries
import requests
import pandas as pd
from zipfile import ZipFile

url = "https://github.com/donadviser/datasets/raw/master/data/kaggle-survey-2018.zip"
#fname = "kaggle-survey-2018.zip"
filename = "multipleChoiceResponses.csv"



import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO

def read_zip_file_to_dataframe(url, filename, **kwargs):
  """
  Reads a file from a zip file downloaded from a URL and returns it as a Pandas DataFrame.

  Args:
      url: The URL of the zip file.
      filename: The name of the file within the zip file to read.
      **kwargs: Additional arguments to be passed to pandas.read_csv().

  Returns:
      A Pandas DataFrame containing the contents of the file.

  Raises:
      ValueError: If the filename is not found in the zip file.
  """

  # Download the zip file
  response = requests.get(url)
  response.raise_for_status()

  # Open the zip file in memory
  with ZipFile(BytesIO(response.content)) as zip_file:
    if filename not in zip_file.namelist():
      raise ValueError(f"File '{filename}' not found in zip file.")

    # Read the file into a DataFrame
    with zip_file.open(filename) as file:
      return pd.read_csv(file, **kwargs)


raw = read_zip_file_to_dataframe(url, filename)

# Print the first 5 rows of the DataFrame
print(raw.head())
raw.columns


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
  return(df_
         .assign(age=df_.Q2.str.slice(0,2))
         .loc[:,'age']
  )

df = tweak_kag(raw)

df.head()