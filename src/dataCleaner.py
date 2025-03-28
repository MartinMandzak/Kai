import pandas as pd


df = pd.read_excel("./data/Z_CE11000_12.xlsx")

SKIP = "# Unassigned"

#columns
quantity = "Quantity"
revenue = "Revenue"

filtered_df = df[~df.apply(
                lambda row: row.astype(str).str.contains(SKIP, case=False).any(),
                axis=1)]


def hash_value(val):
    if pd.notna(val):
        hash_value = sum(ord(char) * (i + 43) for i, char in enumerate(str(val)))
        return int(hash_value)
    return val

filtered_df["CustomerID"] = filtered_df["CustomerID"].apply(hash_value)

def clean_nums(val):
    if pd.notna(val):
        new_val = val.replace(",",".")
        new_val = new_val.split()
        if new_val[0][0] == "-":
            #print(new_val[0])
            return -1 * float(new_val[0][1:]) * 1000
        else:
            #print(new_val[0])
            return float(new_val[0])* 1000
        return val


filtered_df["Revenue"] = filtered_df["Revenue"].apply(
                            clean_nums,
                        )

filtered_df["Quantity"] = filtered_df["Quantity"].apply(
                            clean_nums
                        )

filtered_df = filtered_df[filtered_df[revenue] > 0]
filtered_df = filtered_df[filtered_df[quantity] > 0]
filtered_df = filtered_df[filtered_df['Year'] != 2025]

filtered_df = filtered_df.drop_duplicates()

filtered_df.to_excel("./data/AnonymisedData.xlsx", index=False)
