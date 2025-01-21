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
    
   


      

            
            






