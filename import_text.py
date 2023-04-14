import pandas as pd

class Import_Text:

    # Add the ability to save
    # add the lcocation section
    def __init__(self, name, location=None):
        self.name = name
        self.location = location

    # expand this function as it is limited and single purpose
    def line(self, encoding=None, compression=None, header=None, dtype=None, split=None):
        raw_table = pd.read_table(self.name, compression=compression, encoding=encoding, header=header, dtype=dtype)
        table_columns = raw_table.columns


        # Look into removing this loop
        df = pd.DataFrame()
        for i in range(0, len(table_columns)):
            temp_df = raw_table.iloc[:, i].str.split(split)
            temp_df = pd.DataFrame(temp_df.to_list())
            # Look into removing this hard set size
            temp_df = temp_df[[0, 1]]
            temp_df = temp_df.pivot(columns=0)

            # Another potential loop to remove
            for j in range(0, temp_df.shape[1]):
                temp_column = temp_df.iloc[:, j].dropna().reset_index()[1]
                if temp_column.shape[0] > 0:
                    df[len(df.columns)] = temp_column

        return df
