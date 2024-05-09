import dash
from dash import html, dcc, Input, Output
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Loan Application"),
    html.Div([
        html.Label("Input 1"),
        dcc.Input(id='input1', type='text')
    ]),
    html.Div([
        html.Label("Input 2"),
        dcc.Input(id='input2', type='text')
    ]),
    html.Div([
        html.Label("Input 3"),
        dcc.Input(id='input3', type='text')
    ]),
    html.Div([
        html.Label("Input 4"),
        dcc.Input(id='input4', type='text')
    ]),
    html.Div([
        html.Label("Input 5"),
        dcc.Input(id='input5', type='text')
    ]),
    html.Div([
        html.Label("Input 6"),
        dcc.Input(id='input6', type='text')
    ]),
    html.Div([
        html.Label("Input 7"),
        dcc.Input(id='input7', type='text')
    ]),
    html.Div([
        html.Label("Input 8"),
        dcc.Input(id='input8', type='text')
    ]),
    html.Div([
        html.Label("Input 9"),
        dcc.Input(id='input9', type='text')
    ]),
    html.Div([
        html.Label("Input 10"),
        dcc.Input(id='input10', type='text')
    ]),
    html.Div([
        html.Label("Input 11"),
        dcc.Input(id='input11', type='text')
    ]),
    html.Div([
        html.Label("Input 12"),
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
        df.to_csv('../Machine-Learning-382-Project-1.csv', mode='a', header=False, index=False)
        
        return 'Data added to CSV successfully.'

if __name__ == '__main__':
    app.run_server(debug=True)
