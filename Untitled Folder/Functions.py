import pandas as pd
class load_data:
    def __init__(self,filepath):
        self.filepath=filepath
        self.df=pd.read_csv(self.filepath)
        print("Data loaded successfully.")
              
    def get_data(self):
        return self.df
    

    
def explore_data(df):
    """
    - Shape of the dataset
    - Info (data types, non-null counts)
    - Statistical summary (describe)
    
    """
    # Displaying the first 5 rows
    print("displaying the first 5 rows","\n", df.head(),"\n")
    
    # Display shape
    print("Shape of the dataset: ", df.shape, "\n")
        
    # Display info
    print("Dataset info:\n")
    df.info()
    print("\n")
        
    # Display statistical summary
    print("Statistical summary:\n", df.describe(), "\n")
    
    #Checking for nulls
    nan_percent = (df.isna().sum() / len(df))*100 # total percent of missing values per column
    print("percentage of nulls","\n", nan_percent)
    
    
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_outliers_with_boxplot(data, column_name):
    """
    Creates a boxplot to visualize potential outliers in a specific column.

    Parameters:
    data (pd.df): The dataframe containing the data.
    column_name (str): The column name to visualize outliers for.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.boxplot(x=data[column_name], color='skyblue')  
    plt.title(f'Boxplot of {column_name}')  
    plt.xlabel(column_name)  
    plt.show()  
    
    
def check_and_remove_duplicates(df):
    """
    Checks for duplicate rows in a DataFrame and removes them if any exist.
    
    Parameters:
    df (pd.df): The DataFrame to check for duplicates.
    
    Returns:
    pd.df: The cleaned DataFrame without duplicates.
    """
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        print("Removing duplicates...")
        df = df.drop_duplicates()
        print("Duplicates removed.")
    else:
        print("No duplicates found.")
    
    return df







   


      

            
            






