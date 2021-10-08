import pickle
import pandas as pd
import numpy as np
import webbrowser
import dash
from plotly.subplots import make_subplots
import datetime as dt
import dash_core_components as dcc
import plotly.offline as py
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.tools as tls
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from dash.dependencies import Output, Input
from dash_extensions import Lottie       # pip install dash-extensions
import plotly.express as px              # pip install plotly
import pandas as pd                      # pip install pandas
from datetime import date
import calendar    
from wordcloud import WordCloud,ImageColorGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "COROV-D"
path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/10-02-2021.csv'
url_coonections = "https://assets8.lottiefiles.com/private_files/lf30_vnuqe1wi.json"
url_companies = "https://assets10.lottiefiles.com/packages/lf20_exktvc9d.json"
url_msg_in = "https://assets2.lottiefiles.com/packages/lf20_0r8YYU.json"
url_msg_out = "https://assets9.lottiefiles.com/packages/lf20_culjluaz.json"
url_reactions = "https://assets9.lottiefiles.com/packages/lf20_vewnyqdu.json"
url_coonections1 = "https://assets4.lottiefiles.com/packages/lf20_kpu00she.json"
url_companies1 = "https://assets8.lottiefiles.com/packages/lf20_sb5rlinb.json"
url_msg_in1 = "https://assets8.lottiefiles.com/private_files/lf30_P1NG3T.json"
url_reactions1 = "https://assets3.lottiefiles.com/packages/lf20_74souiny.json"
url_msg_out1 = "https://assets1.lottiefiles.com/private_files/lf30_fxjz1nl5.json"

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#1F2630",
}

def open_browser():
    return webbrowser.open_new("http://127.0.0.1:8050/")

def load_model():
    global pickle_model
    global vocab
    global df, dfs
    
    
    with open("pickle_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    with open("features.pkl", 'rb') as voc:
        vocab = pickle.load(voc)
        
    
        
def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    
    return pickle_model.predict(vectorised_review)


def load_dataset():
    global cdf
    global std
    global df,world
    global tdf,sampling
    df = pd.read_csv(path)
    df.drop(['FIPS', 'Admin2','Last_Update','Province_State', 'Combined_Key'], axis=1, inplace=True)
    df.rename(columns={'Country_Region': "Country"}, inplace=True)
    world = df.groupby("Country")['Confirmed','Active','Recovered','Deaths'].sum().reset_index()
    cdf = pd.read_csv('covid_19_india.csv',parse_dates=['Date'],dayfirst=True)
    cdf = cdf[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
    cdf.columns = ['date','state','cured','deaths','confirmed']
    cdf.state = cdf['state'].str.replace('*','')
    std = pd.read_csv('covid_vaccine_statewise.csv',parse_dates=['Updated On'],dayfirst=True)
    std = std.replace(np.nan, 0)
    tdf = pd.read_csv('vaccineanalysis.csv')
    tdf.drop(['Subjectivity','Polarity'], axis=1, inplace=True)
    tdf = tdf.astype({"text": str})
    sampling = pd.read_csv('StatewiseTestingDetails.csv',parse_dates=['Date'],dayfirst=True)
    sampling = sampling.replace(np.nan, 0)

def create_app_ui():
    global project_name
    global cdf
    global chartdf
    global std
    global today,max_confirmed_cases,max_cured_cases
    global fig,fig2,fig3,figure,fig4,plotly_fig,plotly_fig2,sentifig,accfig
    global allalgos
    allalgos=["DECISION TREE","KNN","RANDOM FOREST","NAIVE BAYES","SVC"]
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split(tdf['text'], tdf['Analysis'], random_state = 42 )
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(min_df = 5).fit(features_train)
    len(vect.get_feature_names())
    vect.get_feature_names()[10000:10010]
    features_train_vectorized = vect.transform(features_train)
    #decision tree
    model = DecisionTreeClassifier()
    model.fit(features_train_vectorized, labels_train)
    predictions = model.predict(vect.transform(features_test))
    decisiontreescore = (f1_score(labels_test, predictions, average=None)*100)
    #knn
    model = KNeighborsClassifier(n_neighbors = 7).fit(features_train_vectorized, labels_train)
    predictions = model.predict(vect.transform(features_test))
    knnscore = (f1_score(labels_test, predictions, average=None)*100)
    #random forest
    model = RandomForestClassifier()
    model.fit(features_train_vectorized, labels_train)
    predictions = model.predict(vect.transform(features_test))
    #score
    rfscore = (f1_score(labels_test, predictions, average=None)*100)
    #Naive Bayes
    model = MultinomialNB().fit(features_train_vectorized, labels_train)
    predictions = model.predict(vect.transform(features_test))
    nbscore = (f1_score(labels_test, predictions, average=None)*100)
    #svm
    model = SVC(kernel = 'linear', C = 1).fit(features_train_vectorized, labels_train)
    predictions = model.predict(vect.transform(features_test))
    svmscore = (f1_score(labels_test, predictions, average=None)*100)
    accfig = go.Figure()
    accfig.add_trace(go.Line(name="DECISION TREE",
                    x = [0,1,2],
                    y = decisiontreescore,
                    marker=dict(
                        color='#FE0000',
                        line=dict(color='#FE0000', width=3)
                    ),
                    ))
    accfig.add_trace(go.Line(name="KNN",
                    x = [0,1,2],
                    y = knnscore,
                    marker=dict(
                        color='#89ACFF',
                        line=dict(color='#89ACFF', width=3)
                    ),
                    ))
    accfig.add_trace(go.Line(name="RANDOM FOREST",
                    x = [0,1,2],
                    y = rfscore,
                    marker=dict(
                        color='#C0809C',
                        line=dict(color='#C0809C', width=3)
                    ),
                    ))
    accfig.add_trace(go.Line(name="NAIVE BAYES",
                    x = [0,1,2],
                    y = nbscore,
                    marker=dict(
                        color='#F786A8',
                        line=dict(color='#F786A8', width=3)
                    ),
                    ))
    accfig.add_trace(go.Line(name="SVC",
                    x = [0,1,2],
                    y = svmscore,
                    marker=dict(
                        color='#73D83C',
                        line=dict(color='#73D83C', width=3)
                    ),
                    ))
    today = cdf[cdf.date == '2020-07-07']
    max_confirmed_cases=today.sort_values(by="confirmed",ascending=False)
    states = max_confirmed_cases.iloc[0:5,1].values
    confirmed = max_confirmed_cases.iloc[0:5,-1].values
    max_cured_cases=today.sort_values(by="cured",ascending=False)
    states2 = max_cured_cases.iloc[0:5,1].values
    cured = max_cured_cases.iloc[0:5,-1].values
    max_deaths_cases=today.sort_values(by="deaths",ascending=False)
    states3 = max_deaths_cases.iloc[0:5,1].values
    deaths = max_deaths_cases.iloc[0:5,-1].values
    mood = tdf["Analysis"].value_counts()
    names = list(mood.index)
    values = list(mood.values)
    allWords = ' '.join( [twts for twts in tdf['text']] )
    mask = np.array(Image.open("injection.png"))
    wc = WordCloud(background_color='white', mask=mask, mode='RGB',
                   width=1000, max_words=1000, height=1000,
                   random_state=1, contour_width=1, contour_color='steelblue')
    wc.generate(allWords)
    wc.to_file("assets/wordCloud.png")
    figure = px.choropleth(world,locations='Country', locationmode='country names', color='Confirmed', hover_name='Country', color_continuous_scale='tealgrn', range_color=[1,1000000],title='Countries with Confirmed cases')
    fig = go.Figure()
    fig.add_trace(go.Bar(x = states,
                    y = confirmed,
                    marker=dict(
                        color='#FE0000',
                        line=dict(color='#FE0000', width=3)
                    ),
                    width=[0.4, 0.4, 0.4, 0.4, 0.4]
                    ))
    fig.update_layout(plot_bgcolor = "#1E1F23",
                    font = dict(color = "#909497"),
                    title = dict(text = "TOP 5 CONFIRMED CASES"),
                    xaxis = dict(title = "States", linecolor = "#909497"),
                    yaxis = dict(title = "Confirmed Cases", linecolor = "#909497"))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x = states2,
                    y = cured,
                    marker=dict(
                        color='#50E2BB',
                        line=dict(color='#50E2BB', width=3)
                    ),
                    width=[0.4, 0.4, 0.4, 0.4, 0.4]
                    ))
    fig2.update_layout(plot_bgcolor = "#1E1F23",
                    font = dict(color = "#909497"),
                    title = dict(text = "TOP 5 CURED CASES"),
                    xaxis = dict(title = "States", linecolor = "#909497"),
                    yaxis = dict(title = "Cured Cases", linecolor = "#909497"))
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x = states3,
                    y = deaths,
                    marker=dict(
                        color='#7ED6FC',
                        line=dict(color='#7ED6FC', width=3)
                    ),
                    width=[0.4, 0.4, 0.4, 0.4, 0.4]
                    ))
    fig3.update_layout(plot_bgcolor = "#1E1F23",
                    font = dict(color = "#909497"),
                    title = dict(text = "TOP 5 DEATHS"),
                    xaxis = dict(title = "States", linecolor = "#909497"),
                    yaxis = dict(title = "Death Cases", linecolor = "#909497"))
    sentifig = go.Figure()
    sentifig.add_trace(go.Bar(x = names,
                    y = values,
                    marker=dict(
                        color='#50E2BB',
                        line=dict(color='#50E2BB', width=3)
                    ),
                    width=[0.4, 0.4, 0.4, 0.4, 0.4]
                    ))
    sentifig.update_layout(plot_bgcolor = "#1E1F23",
                    font = dict(color = "#909497"),
                    title = dict(text = "SENTIMENTS OF COVID VACCINE TWEETS"),
                    xaxis = dict(title = "Sentiments", linecolor = "#909497"),
                    yaxis = dict(title = "Count", linecolor = "#909497"))
    
    sidebar = html.Div([
            dbc.CardImg(src='/assets/logo.png'),
            html.Hr(),
            html.P(
                "Visualise Analyse Forecast", className="lead",style={'color':'white'}),
            dbc.Nav(
                [
                    dbc.NavLink("Dashboard", href="/", active="exact"),
                    dbc.NavLink("Top 5", href="/page-1", active="exact"),
                    dbc.NavLink("World Map", href="/page-2", active="exact"),
                    dbc.NavLink("Forecasting", href="/page-3", active="exact"),
                    dbc.NavLink("Sentiment Analysis of Twitter tweets on Coivd Vaccine", href="/page-4", active="exact"),
                    dbc.NavLink("NLP based ML model for sentiment analysis of COVID Vaccine texts", href="/page-5", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE)
    content = html.Div(id="page-content", children=[])
    main_layout = html.Div([
                dcc.Location(id="url"),
                sidebar,
                content
            ])
    return main_layout
        
# Updating the 5 number cards ******************************************
@app.callback(
    Output('content-connections','children'),
    Output('content-companies','children'),
    Output('content-msg-in','children'),
    Input('my-date-picker-start','date'),
    Input('my-date-picker-end','date'),
    Input('dropdownstate','value')
)
def update_small_cards(start_date, end_date, statevalue):
    # Connections
    dff_c = cdf.copy()
    std_c = std.copy()
    std_c = std_c.replace(np.nan, 0)
    dff_c = dff_c[(dff_c['date']>=start_date) & (dff_c['date']<=end_date) & (dff_c.state == statevalue)]
    val1 = dff_c.iloc[0,4:5].values
    val2 = dff_c.iloc[-1,4:5].values
    val3 = int(val2-val1)
    valdeath1 = dff_c.iloc[0,2:3].values
    valdeath2 = dff_c.iloc[-1,2:3].values
    valdeath3 = int(valdeath2-valdeath1)
    valcured1 = dff_c.iloc[0,3:4].values
    valcured2 = dff_c.iloc[-1,3:4].values
    valcured3 = int(valcured2-valcured1)
    conctns_num = val3
    compns_num = valdeath3
    in_num = valcured3
    return conctns_num, compns_num, in_num

# Updating the 5 number cards ******************************************
@app.callback(
    Output('content-reactions','children'),
    Output('content-connections1','children'),
    Output('content-msg-out','children'),
    Input('my-date-picker-start','date'),
    Input('my-date-picker-end','date'),
    Input('dropdownstate','value')
)
def update_small_cards(start_date, end_date,statevalue):
    # Connections
    samps = sampling.copy()
    std_c = std.copy()  
    samps = samps[(samps['Date']>=start_date) & (samps['Date']<=end_date) & (samps.State == statevalue)]
    valf1 = samps.iloc[0,2:3].values
    valf2 = samps.iloc[-1,2:3].values
    valf3 = int(valf2-valf1)
    conctns_num1 = valf3
    std_c = std_c[(std_c['Updated On']>=start_date) & (std_c['Updated On']<=end_date) & (std_c.State == statevalue)]
    valf1 = std_c.iloc[0,6:7].values
    valf2 = std_c.iloc[-1,6:7].values
    valf3 = int(valf2-valf1)
    compns_num1 = valf3
    std_c = std_c[(std_c['Updated On']>=start_date) & (std_c['Updated On']<=end_date) & (std_c.State == statevalue)]
    valf1 = std_c.iloc[0,5:6].values
    valf2 = std_c.iloc[-1,5:6].values
    valf3 = int(valf2-valf1)
    out_num = valf3


    return out_num,conctns_num1, compns_num1

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                dbc.Container([
        dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('SELECT THE STATE AND THE DATES TO VIEW'),
                    dcc.DatePickerSingle(
                        id='my-date-picker-start',
                        date=pd.to_datetime('2021, 4, 4'),
                        className='ml-5'
                    ),
                    dcc.DatePickerSingle(
                        id='my-date-picker-end',
                        date=pd.to_datetime('2021, 7, 7'),
                        className='mb-2 ml-2'
                    ),
                     dcc.Dropdown(
                    id='dropdownstate',
                    placeholder = 'Select a state',
                    options=[{'label': i, 'value': i} for i in cdf.state.unique()],
                    value=cdf.state[0],
                    style = {'margin-bottom': '10px','font-size': '13px', 'color' : '#000', 'white-space': 'nowrap', 'text-overflow': 'ellipsis','width': '274px','margin-left': '23px'}
                    ),
                ])
            ], color="info", style={'height':'23vh','margin-top':'20px'}),
        ], width=6),
    ],className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_coonections)),
                dbc.CardBody([
                    html.H6('Confirmed Cases'),
                    html.H3(id='content-connections', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_companies)),
                dbc.CardBody([
                    html.H6('Cured Cases'),
                    html.H3(id='content-companies', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_msg_in)),
                dbc.CardBody([
                    html.H6('Deaths'),
                    html.H3(id='content-msg-in', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
    ],className='mb-2'),
        dbc.Row([
                    dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_msg_out)),
                dbc.CardBody([
                    html.H6('First Dose Done'),
                    html.H3(id='content-msg-out', children="000")
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),
                    dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_reactions)),
                dbc.CardBody([
                    html.H6('Second Dose Done'),
                    html.H3(id='content-reactions', children="000")
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_coonections1)),
                dbc.CardBody([
                    html.H6('Covid Samples Collected'),
                    html.H3(id='content-connections1', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
    ],className='mb-2')
], fluid=True,id='mainblock')
                ]
    elif pathname == "/page-1":
        return [dbc.Container([
                dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('VISUALISATIONS OF THE COVID CASES',id="vish3")
                ])
            ], color="success", style={'height':'12vh','margin-top':'15px','margin-bottom':'15px'}),
        ], width=10),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id= 'confirmedgraph', figure=fig),
                ])
            ]),
        ], width=10,style={'margin-bottom':'10px'}),
        dbc.Col([ 
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id= 'curedgraph', figure=fig2),
                ])
            ]),
        ], width=10,style={'margin-bottom':'10px'}),
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                   dcc.Graph(id= 'deathgraph', figure=fig3),
                ])
            ]),
        ], width=10),
    ],className='mb-2')
                ],id='mainblock2')]
    elif pathname == "/page-2":
        return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                               dcc.Graph(id= 'deathgraph', figure=figure),
                            ])
                        ]),
                    ], width=10),
                ],className='mb-2')
                ],id='mainblock3')
    elif pathname=="/page-3":
        return dbc.Container([
        dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('SELECT THE STATE AND THE DATES TO VIEW'),
            dcc.DatePickerSingle(
                        id='preddate',
                        date=pd.to_datetime('2021, 8, 11'),
                        className='mb-2 ml-2'
                    ),
            dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a state',
                    options=[{'label': i, 'value': i} for i in cdf.state.unique()],
                    value=cdf.state[0],
                    style = {'margin-bottom': '10px','font-size': '13px', 'color' : '#000', 'white-space': 'nowrap', 'text-overflow': 'ellipsis','width': '274px','margin-left': '0px'}
                    ),
                ])
            ], color="info", style={'height':'28vh','margin-top':'20px'}),
                    ], width=6),
    ],className='mb-2 mt-2'),
                    dbc.Row([
        dbc.Col([
            dbc.Card([
                        dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_coonections)),
                        dbc.CardBody([
                            html.H6('PREDICTING CONFIRMED CASES'),
                            html.H3(id='predcases', children="000")
                        ], style={'textAlign': 'center'})
                    ]),
                ],width=3),
        dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(Lottie(options=options, width="67%", height="67%", url=url_companies)),
                        dbc.CardBody([
                            html.H6('PREDICTING CURED CASES'),
                            html.H3(id='predcases2', children="000")
                        ], style={'textAlign': 'center'})
                    ]),
                ],width=3)
    ],className='mb-2')
                ],id="mainblock4")
    elif pathname=="/page-4":
        return dbc.Container([
            dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Img(src=app.get_asset_url('wordCloud.png'),style={'width':'400px','height':'455px','margin':'-5px -421px 0 0'}),
                ])
            ]),
        ], width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id= 'confirmedgraph', figure=sentifig),
                ])
            ]),
        ], width=5),
    ],className='mb-2'),
                ],id="mainblock5")
    elif pathname=="/page-5":
        return dbc.Container([
            dbc.Row([
        dbc.Col([
            dbc.Card([
               dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", style = {'height': '345px'}),
               dbc.Button("Check", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
               html.Div(children = None, id='result'),
            ]),
        ], width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id= 'line-chart', figure=accfig),
                ])
            ]),
        ], width=7),
    ],className='mb-2'),
                ],id="mainblock5")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    if (n_clicks > 0):
        response = check_review(textarea)
        if (response[0] == 1):
            return dbc.Alert("Negative", color="danger")
        elif (response[0] == 2 ):
            return dbc.Alert("Neutral", color="info")
        else:
            return dbc.Alert("Positive", color="success")
    else:
        return ""

@app.callback(
    Output('predcases','children'),
    Output('predcases2','children'),
    Input('preddate','date'),
    Input('dropdown','value')
    )
def update_pred(datetopred,statevalue):
    global features,labels,fig11
    maha = cdf[cdf.state == statevalue]
    maha['date']=maha['date'].map(dt.datetime.toordinal)
    features=maha.iloc[:,0:1].values
    labels=maha.iloc[:,4:5].values
    labels2=maha.iloc[:,2].values
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(features, labels)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    higher_degree_gen = PolynomialFeatures(degree = 2)
    features_poly = higher_degree_gen.fit_transform(features)
    regressor_poly = LinearRegression()
    regressor_poly.fit(features_poly, labels)
    datetopred = pd.to_datetime(datetopred)
    today = datetopred.toordinal()
    x=[today]
    x=np.array(x)
    x=x.reshape(1,1)
    x=higher_degree_gen.transform(x)
    pred1=regressor_poly.predict(x)
    higher_degree_gen = PolynomialFeatures(degree = 2)
    features_poly = higher_degree_gen.fit_transform(features)
    regressor_poly = LinearRegression()
    regressor_poly.fit(features_poly, labels2)
    datetopred = pd.to_datetime(datetopred)
    today = datetopred.toordinal()
    x=[today]
    x=np.array(x)
    x=x.reshape(1,1)
    x=higher_degree_gen.transform(x)
    pred2=regressor_poly.predict(x)
    return int(pred1),int(pred2)

def main():
    global app
    global project_name
    load_model()
    load_dataset()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None
if __name__ == '__main__':
    main()