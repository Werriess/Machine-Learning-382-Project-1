import dash
from dash import html, dcc, Input, Output
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.prepare_data2 import prepare_data

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Loan Application"),
    html.Div([
        html.Label("Loan_ID"),
        dcc.Input(id='input1', type='text')
    ]),
    html.Div([
        html.Label("Gender"),
        dcc.Input(id='input2', type='text')
    ]),
    html.Div([
        html.Label("Married"),
        dcc.Input(id='input3', type='text')
    ]),
    html.Div([
        html.Label("Dependents"),
        dcc.Input(id='input4', type='text')
    ]),
    html.Div([
        html.Label("Education"),
        dcc.Input(id='input5', type='text')
    ]),
    html.Div([
        html.Label("Self_Employed"),
        dcc.Input(id='input6', type='text')
    ]),
    html.Div([
        html.Label("ApplicantIncome"),
        dcc.Input(id='input7', type='text')
    ]),
    html.Div([
        html.Label("CoapplicantIncome"),
        dcc.Input(id='input8', type='text')
    ]),
    html.Div([
        html.Label("LoanAmount"),
        dcc.Input(id='input9', type='text')
    ]),
    html.Div([
        html.Label("Loan_Amount_Term"),
        dcc.Input(id='input10', type='text')
    ]),
    html.Div([
        html.Label("Credit_History"),
        dcc.Input(id='input11', type='text')
    ]),
    html.Div([
        html.Label("Property_Area"),
        dcc.Input(id='input12', type='text')
    ]),
    
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='output-div')
])

# Callback to handle button click and write to CSV
@app.callback(
    Output('output-div', 'children'),
    [Input('submit-val', 'n_clicks')],
    [Input('input{}'.format(i), 'value') for i in range(1, 13)]
)
def update_output(n_clicks, *inputs):
    if n_clicks > 0:
        # Create a DataFrame with the input data
        data = {'Input {}'.format(i): [val] for i, val in enumerate(inputs, start=1)}
        df = pd.DataFrame(data)
        
        # Append data to CSV file
        df.to_csv('../MLG382 Projects/Machine-Learning-382-Project-1/data/loan_applications.csv', mode='a', header=False, index=False)
        
        return make_prediction()
    
def make_prediction():
    prediction_data = prepare_data('../MLG382 Projects/Machine-Learning-382-Project-1/data/loan_applications.csv')
    model = joblib.load('../MLG382 Projects/Machine-Learning-382-Project-1/artifacts/model1.pkl')
    
    scaler = StandardScaler()
    scaled_input = scaler.transform(prediction_data)
    
    prediction = model.predict(scaled_input)
    
    return prediction

if __name__ == '__main__':
    app.run_server(debug=True)
