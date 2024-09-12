import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

data = pd.read_csv('Stock_Prediction.csv') 
originaldata = data[['Date', 'Store_Name','Product_Category','Stock_Sold','Sales']]
data['Date'] = pd.to_datetime(data['Date'], format='%m-%Y')

multipliers = data.groupby('Product_Category').apply(lambda x: x['Sales'].sum() / x['Stock_Sold'].sum()).to_dict()

stores = data['Store_Name'].unique()
categories = data['Product_Category'].unique()

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
pipeline.model.eval()
results = []

for store in stores:
    for category in categories:
        subset = data[(data['Store_Name'] == store) & (data['Product_Category'] == category)]
        subset = subset[['Date', 'Stock_Sold']].sort_values('Date').reset_index(drop=True)

        if subset.empty:
            continue

        context = torch.tensor(subset['Stock_Sold'].values, dtype=torch.float32).unsqueeze(0) 

        prediction_length = 36
        forecast = pipeline.predict(context, prediction_length)  

        last_date = subset['Date'].iloc[-1]
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, prediction_length + 1)]

        high = np.quantile(forecast[0].numpy(), 0.7, axis=0)
        high = np.round(high).astype(int)
        future_dates_str = [date.strftime('%m-%Y') for date in future_dates]

        sales = high * multipliers.get(category, 1) 

        forecast_df = pd.DataFrame({
            'Date': future_dates_str,
            'Store_Name': store,
            'Product_Category': category,
            'Stock_Sold': high.astype(int),
            'Sales': sales.astype(int) 
        })

        results.append(forecast_df)

all_forecasts = pd.concat(results, ignore_index=True)

def filter_data_by_store(df,store_name):
    temp = df[df['Store_Name'] == store_name]
    return temp

def filter_data_by_category(df,Product_Category):
    temp = df[df['Product_Category'] == Product_Category]
    return temp

def filter_data_by_year_month(df,datetime):
    month = datetime.split('-')[0]
    year = datetime.split('-')[1]
    filtered_df = df
    filtered_df['Date'] = filtered_df['Date'].astype(str)
    mask = pd.Series([True] * len(filtered_df))
    if month is not None and month.strip() != "":
        month = int(month)
        if 1 <= month <= 12:
            mask &= df['Date'].str[:2].astype(int) == month
        else:
            raise ValueError("Month must be between 1 and 12.")

    if year is not None and year.strip() != "":
        year = int(year)
        mask &= df['Date'].str[3:].astype(int) == year

    filtered_df = filtered_df[mask]
    return filtered_df

from flask import Flask, request, json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/get_forecasted_data', methods=['GET'])
def get_forecasted_data():
    prompt = request.args.get('prompt')
    if prompt is None or prompt == "":
        return all_forecasts.to_json(orient='records', lines=False, indent=4)    
    else:
        if prompt == "existing_data":
            return originaldata.to_json(orient='records', lines=False, indent=4)
        else:
            if prompt == "combined_data":
                combined_data = pd.concat([originaldata, all_forecasts], axis=0)
                return combined_data.to_json(orient='records', lines=False, indent=4)
            else:
                split_prompt = prompt.split('|')
                filtered_forecast = all_forecasts
                if split_prompt[0] is not None and  split_prompt[0].strip() != "":
                    filtered_forecast = filter_data_by_year_month(filtered_forecast,split_prompt[0])
                if split_prompt[1] is not None and  split_prompt[1].strip() != "":
                    filtered_forecast = filter_data_by_store(filtered_forecast,split_prompt[1])
                if split_prompt[2] is not None and  split_prompt[2].strip() != "":
                    filtered_forecast = filter_data_by_category(filtered_forecast,split_prompt[2])
                return filtered_forecast.to_json(orient='records', lines=False, indent=4)


@app.route('/get_combined_data', methods=['GET'])
def get_combined_data():
    prompt = request.args.get('prompt')
    if prompt is None or prompt == "":
        return all_forecasts.to_json(orient='records', lines=False, indent=4)    
    else:
        if prompt == "existing_data":
            return originaldata.to_json(orient='records', lines=False, indent=4)
        else:
            if prompt == "combined_data":
                combined_data = pd.concat([originaldata, all_forecasts], axis=0)
                return combined_data.to_json(orient='records', lines=False, indent=4)
            else:
                split_prompt = prompt.split('|')
                filtered_forecast = pd.concat([originaldata, all_forecasts], ignore_index=True)
                print(filtered_forecast)
                if split_prompt[0] is not None and  split_prompt[0].strip() != "":
                    filtered_forecast = filter_data_by_year_month(filtered_forecast,split_prompt[0])
                if split_prompt[1] is not None and  split_prompt[1].strip() != "":
                    filtered_forecast = filter_data_by_store(filtered_forecast,split_prompt[1])
                if split_prompt[2] is not None and  split_prompt[2].strip() != "":
                    filtered_forecast = filter_data_by_category(filtered_forecast,split_prompt[2])
                return filtered_forecast.to_json(orient='records', lines=False, indent=4)


if __name__ == '__main__':
  app.run(host='0.0.0.0')
