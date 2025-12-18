import pandas as pd


df = pd.read_excel("./data/Z_CE11000_12.xlsx")

SKIP = "# Unassigned" # this was like half the document, the export from SAP wasn't ideal

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

# this is me swapping decimal , for an actual .
'''
Now, for the actual more fun parts of clean_nums
First, the easy part > the values were written like 1,34 Thousands.. so I replaced the comma with a dot and multiplied by a 1000
NOW
I mentioned at the interview that there was a lot of internal transactions in regards to warehouse movement.
It was generally subtracted from one warehouse and added into the another, same value ( this will be important later)
so I check every single negative value within the document and multiply it by -1, making it a positive value instead
this will become important near the end of the code
'''
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
#decided to exclude the year 2025 since the data ended at around April of that year
filtered_df = filtered_df[filtered_df['Year'] != 2025]

# Remember when we multiplied all negative transactions which were just substractions from warehouse A so it can get added to warehouse B?
# Well that created lots of duplicates, where even things like an invoice number were the same, and we can get rid of those duplicates pretty easily!
# Was there a better way of getting rid of those transactions? PROBABLY! I couldn't figure one out though...
filtered_df = filtered_df.drop_duplicates(keep=False)

filtered_df.to_excel("./data/AnonymisedData.xlsx", index=False)
