import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from keras.models import Model, load_model
import streamlit as st
import datetime
import time
import plotly.graph_objects as go
current_time = datetime.datetime.now()

st.title('e-Curriences, Stocks & Soukz')
st.sidebar.title('Select Trading Option')
st.sidebar.markdown("Select option accordingly:")
selected_status = st.sidebar.selectbox('Select Options', options = ['Traditional Method', 'Regression', 'Screener & Indicators', 'About'])

start = '2010-01-01'
end = current_time


if selected_status == 'Traditional Method':

    original_list = ['Stock', 'Real Time Price', 'Sector Wise Stock', 'Crypto Currency']
    result = st.selectbox('Select your trading option', original_list)
    st.write(f'selected trading option: ',(result))
    if result == 'Stock':
        st.subheader('Stock Trend Prediction')
        user_input = st.text_input('Enter Stock Ticker', 'AAPL')


        from pandas_datareader import data as pdata
        import fix_yahoo_finance  # noqa
        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input, start, end)


        #describing Date 
        st.subheader('Data from 2010 - CurrentTime')
        st.write(df.describe())

        #visualization 
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize= (12,6))
        plt.plot(df.Close)
        st.pyplot(fig)



        st.subheader('Closing Price vs Time Chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize= (12,6))
        plt.plot(ma100)
        plt.plot(df.Close)
        st.pyplot(fig)




        st.subheader('Closing Price vs Time Chart with 100MA. & 365MA')
        ma100 = df.Close.rolling(100).mean()
        ma365 = df.Close.rolling(365).mean()
        fig = plt.figure(figsize= (12,6))
        plt.plot(ma100, 'r')
        plt.plot(ma365, 'b')
        plt.plot(df.Close, 'g')
        st.pyplot(fig)



        # splitting data into training and testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])



        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))

        data_training_array = scaler.fit_transform(data_training)





        #load my model 
        model = load_model('keras_model.h5')



        #testing part 

        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor




        #final graph 

        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label = 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

        st.header('Comparisons Between Most Active Stocks')
        st.subheader('(Note : These are most active companies/stocks.)')
        import yfinance as yf

        tickers = ('TTM', 'TCS.NS', 'IDEA.NS', 'ZOMATO.NS', 'UVSL.NS', 'UVSL.NS', 'IDEA.BO', 'VISESHINFO.NS', 'VIKASLIFE.NS', 'SAIL.NS', 'YESBANK.NS', 'VIKASLIFE.BO', 'FEDERALBNK.NS', 'VISESHINFO.BO', 'NMDC.NS', 'VIKASECO.BO', 'JPPOWER.NS', 'ZOMATO.BO', 'ITC.NS', 'VLAKSHVILAS.NS' , 'PNB.NS', 'IDFCFIRSTB.NS', 'SOUTHBANK.NS', 'TATAMOTORS.NS', 'BHARTIARTL.NS', 'IOB.NS', 'SUZLON.NS', 'TVSMOTOR-BL.NS', '^NSEI', 'SBIN.NS', 'AXS', 'AXISBANK.NS', 'TATASTEEL.NS', 'TATAPOWER.NS', 'IBN', 'ICICIBANK.NS', 'RELI', ' GILD', ' UNP', ' UTX', ' HPQ', 'V', 'CSCO', 'AMGN', 'BA', 'COP', 'CMCSA', 'BMY', 'VZ', 'T', 'UNH', 'MCD', 'PFE', 'ABT', 'FB', 'DIS', 'MMM', 'ORCL', 'PEP', 'HD', 'JPM', 'INTC', 'WFC', 'MRK', 'KO', 'AMZN', 'PG', ' BRK.B', 'GOOGL', 'WMT', 'CVX', 'JNJ', 'MO', 'IBM', 'GE', 'MSFT', 'AAPL', 'XOM')

        dropdown = st.multiselect('pick your assets', tickers)

        starts = st.date_input('Start', value = pd.to_datetime('2010-01-01'))
        ends = st.date_input('End', value = pd.to_datetime('today'))

        def relativeret(df):
            rel = df.pct_change()
            cumret = (1+rel).cumprod() - 1
            cumret = cumret.fillna(0)
            return cumret


        if len(dropdown) > 0:
            #df yf.download(dropdown,starts,ends)['Adj Close']
            df = relativeret(yf.download(dropdown,starts,ends)['Adj Close'])
            st.header('Returns of {}'.format(dropdown))
            st.line_chart(df)

    if result == 'Real Time Price':
        st.subheader('Real Time Price')
        import yfinance as yf

        tickers = ('^BSESN', '^NSEI','^DJI', '^IXIC', 'BTC-INR', '^CMC200', '^HSI', '^N225', 'EURINR=X', 'GBPINR=X', 'AEDINR=X', 'INRJPY=X', 'SGDINR=X','(ES=F', 'CL=F', 'GC=F', 'SI=F')
        start = '2010-01-01'
        current_time = datetime.datetime.now()
        end = current_time
        st.subheader('data from 2010-01-01 to today')

        dropdown = st.multiselect('pick your assets', tickers)

        def relativeret(df):
            rel = df.pct_change()
            cumret = (1+rel).cumprod()
            cumret = cumret.fillna(0)
            return cumret

        if len(dropdown) > 0:
            #df yf.download(dropdown,start,end)['Adj Close']
            df = relativeret(yf.download(dropdown,start,end)['Adj Close'])
            st.header('Returns of {}'.format(dropdown))
            st.line_chart(df)

    import yfinance as yf
    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import pandas_datareader as data 
    from keras.models import Model, load_model
    import streamlit as st
    import datetime
    import time
    current_time = datetime.datetime.now()
    start = '2010-01-01'
    end = current_time
    if result == 'Sector Wise Stock':
        st.subheader('Sector Wise Stock Prediction')

        user_input = st.text_input('Enter Stock Sector Ticker', 'XLC')


        from pandas_datareader import data as pdata
        import fix_yahoo_finance  # noqa
        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input, start, end)


        #describing Date 
        st.subheader('Data from 2010 - CurrentTime')
        st.write(df.describe())

          #visualization 
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize= (12,6))
        plt.plot(df.Close)
        st.pyplot(fig)



        st.subheader('Closing Price vs Time Chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize= (12,6))
        plt.plot(ma100)
        plt.plot(df.Close)
        st.pyplot(fig)




        st.subheader('Closing Price vs Time Chart with 100MA. & 365MA')
        ma100 = df.Close.rolling(100).mean()
        ma365 = df.Close.rolling(365).mean()
        fig = plt.figure(figsize= (12,6))
        plt.plot(ma100, 'r')
        plt.plot(ma365, 'b')
        plt.plot(df.Close, 'g')
        st.pyplot(fig)

        # splitting data into training and testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])



        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))

        data_training_array = scaler.fit_transform(data_training)





        #load my model 
        model = load_model('s_s.h5')



        #testing part 

        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor




        #final graph 

        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label = 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

        st.subheader('Stock Sector Wise Comparisons')
        tickers = ('XLC', 'XLU', 'XLP', 'XLV', 'XLK', 'XLRE', 'XLY', 'XLB', 'XLI', 'XLF', 'XLE')
        start = '2010-01-01'
        current_time = datetime.datetime.now()
        end = current_time
        st.subheader('Todays Returns')

        dropdown = st.multiselect('pick your assets', tickers)

        def relativeret(df):
            rel = df.pct_change()
            cumret = (1+rel).cumprod()
            cumret = cumret.fillna(0)
            return cumret

        if len(dropdown) > 0:
            #df yf.download(dropdown,starts,ends)['Adj Close']
            df = relativeret(yf.download(dropdown,start,end)['Adj Close'])
            st.header('Returns of {}'.format(dropdown))
            st.line_chart(df)

    import yfinance as yf
    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import pandas_datareader as data 
    from keras.models import Model, load_model
    import streamlit as st
    import datetime
    import time


    current_time = datetime.datetime.now()
    start = '2012-01-01'
    end = current_time
    if result == 'Crypto Currency':
        st.subheader('Crypto Currency Trend Predictions')
        user_input = st.text_input('Enter Crypto Ticker', 'BTC-INR')


        from pandas_datareader import data as pdata
        import fix_yahoo_finance  # noqa
        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input, start, end)


        #describing Date 
        st.subheader('Data from 2010 - CurrentTime')
        st.write(df.describe())

        #visualization 
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize= (12,6))
        plt.plot(df.Close)
        st.pyplot(fig)



        st.subheader('Closing Price vs Time Chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize= (12,6))
        plt.plot(ma100)
        plt.plot(df.Close)
        st.pyplot(fig)




        st.subheader('Closing Price vs Time Chart with 100MA. & 365MA')
        ma100 = df.Close.rolling(100).mean()
        ma365 = df.Close.rolling(365).mean()
        fig = plt.figure(figsize= (12,6))
        plt.plot(ma100, 'r')
        plt.plot(ma365, 'b')
        plt.plot(df.Close, 'g')
        st.pyplot(fig)



        # splitting data into training and testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])



        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))

        data_training_array = scaler.fit_transform(data_training)





        #load my model 
        model = load_model('c_c.h5')



        #testing part 

        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor




        #final graph 

        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label = 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import yfinance as yf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
import datetime
import time 
current_time = datetime.datetime.now()

start = '2010-01-01'
end = current_time

if selected_status == 'Regression':
    original_list = ['Stock Price', 'Real Time Price', 'Sector Wise Stock', 'Crypto Currency']
    result = st.selectbox('Select your trading option', original_list)
    st.write(f'selected trading option: ',(result))
    if result == 'Stock Price':

        st.subheader('Regression For Stocks')
        user_input = st.text_input('Enter Stock Ticker', 'TSLA')

        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        import pandas as pd
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input)


        df = df[['Adj Close']]

        forecast_len=3
        df['Predicted'] = df[['Adj Close']].shift(-forecast_len)

        x=np.array(df.drop(['Predicted'],1))

        x=x[:-forecast_len]

        y=np.array(df['Predicted'])
        y=y[:-forecast_len]

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

        svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
        svr_rbf.fit(x_train,y_train)


        svr_rbf_confidence=svr_rbf.score(x_test,y_test)
       

        

         #final graph 
        st.header('SVR')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,svr_rbf.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)

        st.info(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")


        lr=LinearRegression()
        lr.fit(x_train,y_train)

        lr_confidence=lr.score(x_test,y_test)
        


        

         #final graph 
        st.header('Linear Regression')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,lr.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)

        st.info(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")
    
    elif result == 'Real Time Price':
        st.subheader('Regression For Real time Price')
        user_input = st.text_input('Enter Price Ticker', 'AEDINR=X')

        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input)


        df = df[['Adj Close']]

        forecast_len=3
        df['Predicted'] = df[['Adj Close']].shift(-forecast_len)

        x=np.array(df.drop(['Predicted'],1))

        x=x[:-forecast_len]

        y=np.array(df['Predicted'])
        y=y[:-forecast_len]

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

        svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
        svr_rbf.fit(x_train,y_train)


        svr_rbf_confidence=svr_rbf.score(x_test,y_test)
         #final graph 
        st.header('SVR')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,svr_rbf.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)

        st.info(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")



        lr=LinearRegression()
        lr.fit(x_train,y_train)

        lr_confidence=lr.score(x_test,y_test)
         #final graph 
        st.header('Linear Regression')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,lr.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)
        st.info(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")

    elif result == 'Sector Wise Stock':
        st.subheader('Regression For Sector Wise Stock')
        user_input = st.text_input('Enter Sector Stock Ticker', 'XLV')

        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input)


        df = df[['Adj Close']]

        forecast_len=3
        df['Predicted'] = df[['Adj Close']].shift(-forecast_len)

        x=np.array(df.drop(['Predicted'],1))

        x=x[:-forecast_len]

        y=np.array(df['Predicted'])
        y=y[:-forecast_len]

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

        svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
        svr_rbf.fit(x_train,y_train)


        svr_rbf_confidence=svr_rbf.score(x_test,y_test)
         #final graph 
        st.header('SVR')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,svr_rbf.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)

        st.info(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")



        lr=LinearRegression()
        lr.fit(x_train,y_train)

        lr_confidence=lr.score(x_test,y_test)
         #final graph 
        st.header('Linear Regression')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,lr.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)
        st.info(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")
    
    elif result == 'Crypto Currency':
        st.subheader('Regression For Crypto Curency')
        user_input = st.text_input('Enter Crypto Currency Ticker', 'LTC-INR')

        import pandas
        from pandas_datareader import data as pdr
        import yfinance as yfin
        yfin.pdr_override()

        df = pdr.get_data_yahoo(user_input)


        df = df[['Adj Close']]

        forecast_len=3
        df['Predicted'] = df[['Adj Close']].shift(-forecast_len)

        x=np.array(df.drop(['Predicted'],1))

        x=x[:-forecast_len]

        y=np.array(df['Predicted'])
        y=y[:-forecast_len]

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

        svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
        svr_rbf.fit(x_train,y_train)


        svr_rbf_confidence=svr_rbf.score(x_test,y_test)
         #final graph 
        st.header('SVR')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,svr_rbf.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)

        st.info(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")



        lr=LinearRegression()
        lr.fit(x_train,y_train)

        lr_confidence=lr.score(x_test,y_test)
         #final graph 
        st.header('Linear Regression')
        
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.scatter(x,y, color='red', label = 'Original Price')
        plt.plot(x,lr.predict(x), color= 'blue' , label = 'Predicted Price')
        plt.xlabel('independent')
        plt.ylabel('dependent')
        plt.legend()
        st.pyplot(fig2)
        st.info(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")



if selected_status == 'Screener & Indicators':
    original_list = ['Stock Price', 'Real Time Price', 'Sector Wise Stock', 'Crypto Currency']
    infoType = st.radio(
        "Choose an info type",
        ('Fundamental', 'Technical')
    ) 
    if(infoType == 'Fundamental'):
        pass
    else:
        pass
    result = st.selectbox('Select your trading option', original_list)
    st.write(f'selected trading option: ',(result))
    if result == 'Stock Price':
        st.subheader('Stocks')
        user_input = st.text_input('Enter Stock Ticker', 'MSFT')
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import datetime as dt
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        if(infoType == 'Fundamental'):
            stock = yf.Ticker(user_input)
            info = stock.info 
            st.title('Company Profile')
            st.subheader(info['longName']) 
            st.markdown('** Sector **: ' + info['sector'])
            st.markdown('** Industry **: ' + info['industry'])
            st.markdown('** Phone **: ' + info['phone'])
            st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
            st.markdown('** Website **: ' + info['website'])
            st.markdown('** Business Summary **')
            st.info(info['longBusinessSummary'])
            fundInfo = {
                'Enterprise Value (USD)': info['enterpriseValue'],
                'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
                'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
                'Net Income (USD)': info['netIncomeToCommon'],
                'Profit Margin Ratio': info['profitMargins'],
                'Forward PE Ratio': info['forwardPE'],
                'PEG Ratio': info['pegRatio'],
                'Price to Book Ratio': info['priceToBook'],
                'Forward EPS (USD)': info['forwardEps'],
                'Beta ': info['beta'],
                'Book Value (USD)': info['bookValue'],
                'Dividend Rate (%)': info['dividendRate'], 
                'Dividend Yield (%)': info['dividendYield'],
                'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
                'Payout Ratio': info['payoutRatio']
            }
    
            fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
            fundDF = fundDF.rename(columns={0: 'Value'})
            st.subheader('Fundamental Info') 
            st.table(fundDF)


            st.subheader('General Stock Info') 
            st.markdown('** Market **: ' + info['market'])
            st.markdown('** Exchange **: ' + info['exchange'])
            st.markdown('** Quote Type **: ' + info['quoteType'])
    
            start = dt.datetime.today()-dt.timedelta(5 * 365)
            end = dt.datetime.today()
            df = yf.download(user_input,start,end)
            df = df.reset_index()
            fig = go.Figure(
                data=go.Scatter(x=df['Date'], y=df['Adj Close'])
                )
            fig.update_layout(
                title={
                    'text': "Stock Prices Over Past Five Years",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig, use_container_width=True)

            marketInfo = {
                "Volume": info['volume'],
                "Average Volume": info['averageVolume'],
                "Market Cap": info["marketCap"],
                "Float Shares": info['floatShares'],
                "Regular Market Price (USD)": info['regularMarketPrice'],
                'Bid Size': info['bidSize'],
                'Ask Size': info['askSize'],
                "Share Short": info['sharesShort'],
                'Short Ratio': info['shortRatio'],
                'Share Outstanding': info['sharesOutstanding']
            }
    
            marketDF = pd.DataFrame(data=marketInfo, index=[0])
            st.table(marketDF)

        else:
            def calcMovingAverage(data, size):
                df = data.copy()
                df['sma'] = df['Adj Close'].rolling(size).mean()
                df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
                df.dropna(inplace=True)
                return df
            def calc_macd(data):
                df = data.copy()
                df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
                df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
                df['macd'] = df['ema12'] - df['ema26']
                df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
                df.dropna(inplace=True)
                return df
            def calcBollinger(data, size):
                df = data.copy()
                df["sma"] = df['Adj Close'].rolling(size).mean()
                df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["width"] = df["bolu"] - df["bold"]
                df.dropna(inplace=True)
                return df
            st.title('Technical Indicators')
            st.subheader('Moving Average')
           

            start = dt.datetime.today()-dt.timedelta(5 * 365)
            end = dt.datetime.today()
            dataMA = yf.download(user_input,start,end)
            df_ma = calcMovingAverage(dataMA, 20)
            df_ma = df_ma.reset_index()
            figMA = go.Figure()
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['Adj Close'],
                        name = "Prices Over Last " + str(5) + " Year(s)"
                    )
                )    
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            figMA.update_layout(legend_title_text='Trend')
            figMA.update_yaxes(tickprefix="inr")
            st.plotly_chart(figMA, use_container_width=True)


            st.subheader('Moving Average Convergence Divergence (MACD)')
            
    
            startMACD = dt.datetime.today()-dt.timedelta(4 * 365)
            endMACD = dt.datetime.today()
            dataMACD = yf.download(user_input,startMACD,endMACD)
            df_macd = calc_macd(dataMACD)
            df_macd = df_macd.reset_index()
    
            figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['Adj Close'],
                        name = "Prices Over Last " + str(4) + " Year(s)"
                    ),
                row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema12'],
                        name = "EMA 12 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema26'],
                        name = "EMA 26 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['macd'],
                        name = "MACD Line"
                ),
            row=2, col=1
            )
    
            figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
            )
    
            figMACD.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                ))
    
            figMACD.update_yaxes(tickprefix="inrs")
            st.plotly_chart(figMACD, use_container_width=True)

            st.subheader('Bollinger Band')

            startBoll= dt.datetime.today()-dt.timedelta(5 * 365)
            endBoll = dt.datetime.today()
            dataBoll = yf.download(user_input,startBoll,endBoll)
            df_boll = calcBollinger(dataBoll, 20)
            df_boll = df_boll.reset_index()
            figBoll = go.Figure()
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bolu'],
                            name = "Upper Band"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['sma'],
                            name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bold'],
                            name = "Lower Band"
                        )
                )
    
            figBoll.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0
            ))
    
            figBoll.update_yaxes(tickprefix="$")
            st.plotly_chart(figBoll, use_container_width=True)


    if result == 'Real Time Price':
        st.subheader('Real TIme Price')
        user_input = st.text_input('Enter Stock Ticker', 'CL=F')
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import datetime as dt
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        if(infoType == 'Fundamental'):
            st.header('Not Available For Real Time Price. Check Technical Analysis')

        else:
            def calcMovingAverage(data, size):
                df = data.copy()
                df['sma'] = df['Adj Close'].rolling(size).mean()
                df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
                df.dropna(inplace=True)
                return df
            def calc_macd(data):
                df = data.copy()
                df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
                df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
                df['macd'] = df['ema12'] - df['ema26']
                df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
                df.dropna(inplace=True)
                return df
            def calcBollinger(data, size):
                df = data.copy()
                df["sma"] = df['Adj Close'].rolling(size).mean()
                df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["width"] = df["bolu"] - df["bold"]
                df.dropna(inplace=True)
                return df
            st.title('Technical Indicators')
            st.subheader('Moving Average')
           

            start = dt.datetime.today()-dt.timedelta(5 * 365)
            end = dt.datetime.today()
            dataMA = yf.download(user_input,start,end)
            df_ma = calcMovingAverage(dataMA, 20)
            df_ma = df_ma.reset_index()
            figMA = go.Figure()
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['Adj Close'],
                        name = "Prices Over Last " + str(5) + " Year(s)"
                    )
                )    
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            figMA.update_layout(legend_title_text='Trend')
            figMA.update_yaxes(tickprefix="inr")
            st.plotly_chart(figMA, use_container_width=True)


            st.subheader('Moving Average Convergence Divergence (MACD)')
            
    
            startMACD = dt.datetime.today()-dt.timedelta(4 * 365)
            endMACD = dt.datetime.today()
            dataMACD = yf.download(user_input,startMACD,endMACD)
            df_macd = calc_macd(dataMACD)
            df_macd = df_macd.reset_index()
    
            figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['Adj Close'],
                        name = "Prices Over Last " + str(4) + " Year(s)"
                    ),
                row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema12'],
                        name = "EMA 12 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema26'],
                        name = "EMA 26 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['macd'],
                        name = "MACD Line"
                ),
            row=2, col=1
            )
    
            figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
            )
    
            figMACD.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                ))
    
            figMACD.update_yaxes(tickprefix="inrs")
            st.plotly_chart(figMACD, use_container_width=True)

            st.subheader('Bollinger Band')

            startBoll= dt.datetime.today()-dt.timedelta(5 * 365)
            endBoll = dt.datetime.today()
            dataBoll = yf.download(user_input,startBoll,endBoll)
            df_boll = calcBollinger(dataBoll, 20)
            df_boll = df_boll.reset_index()
            figBoll = go.Figure()
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bolu'],
                            name = "Upper Band"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['sma'],
                            name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bold'],
                            name = "Lower Band"
                        )
                )
    
            figBoll.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0
            ))
    
            figBoll.update_yaxes(tickprefix="$")
            st.plotly_chart(figBoll, use_container_width=True)

    if result == 'Sector Wise Stock':
        st.subheader('Sector Wise Stock')
        user_input = st.text_input('Enter Stock Ticker', 'XLF')
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import datetime as dt
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        if(infoType == 'Fundamental'):
            st.header('Not Available For Sector wise Stock. Check Technical Analysis')
        else:
            def calcMovingAverage(data, size):
                df = data.copy()
                df['sma'] = df['Adj Close'].rolling(size).mean()
                df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
                df.dropna(inplace=True)
                return df
            def calc_macd(data):
                df = data.copy()
                df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
                df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
                df['macd'] = df['ema12'] - df['ema26']
                df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
                df.dropna(inplace=True)
                return df
            def calcBollinger(data, size):
                df = data.copy()
                df["sma"] = df['Adj Close'].rolling(size).mean()
                df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["width"] = df["bolu"] - df["bold"]
                df.dropna(inplace=True)
                return df
            st.title('Technical Indicators')
            st.subheader('Moving Average')
           

            start = dt.datetime.today()-dt.timedelta(5 * 365)
            end = dt.datetime.today()
            dataMA = yf.download(user_input,start,end)
            df_ma = calcMovingAverage(dataMA, 20)
            df_ma = df_ma.reset_index()
            figMA = go.Figure()
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['Adj Close'],
                        name = "Prices Over Last " + str(5) + " Year(s)"
                    )
                )    
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            figMA.update_layout(legend_title_text='Trend')
            figMA.update_yaxes(tickprefix="inr")
            st.plotly_chart(figMA, use_container_width=True)


            st.subheader('Moving Average Convergence Divergence (MACD)')
            
    
            startMACD = dt.datetime.today()-dt.timedelta(4 * 365)
            endMACD = dt.datetime.today()
            dataMACD = yf.download(user_input,startMACD,endMACD)
            df_macd = calc_macd(dataMACD)
            df_macd = df_macd.reset_index()
    
            figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['Adj Close'],
                        name = "Prices Over Last " + str(4) + " Year(s)"
                    ),
                row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema12'],
                        name = "EMA 12 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema26'],
                        name = "EMA 26 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['macd'],
                        name = "MACD Line"
                ),
            row=2, col=1
            )
    
            figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
            )
    
            figMACD.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                ))
    
            figMACD.update_yaxes(tickprefix="inrs")
            st.plotly_chart(figMACD, use_container_width=True)

            st.subheader('Bollinger Band')

            startBoll= dt.datetime.today()-dt.timedelta(5 * 365)
            endBoll = dt.datetime.today()
            dataBoll = yf.download(user_input,startBoll,endBoll)
            df_boll = calcBollinger(dataBoll, 20)
            df_boll = df_boll.reset_index()
            figBoll = go.Figure()
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bolu'],
                            name = "Upper Band"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['sma'],
                            name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bold'],
                            name = "Lower Band"
                        )
                )
    
            figBoll.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0
            ))
    
            figBoll.update_yaxes(tickprefix="$")
            st.plotly_chart(figBoll, use_container_width=True)


    if result == 'Crypto Currency':
        st.subheader('Crypto Currency')
        user_input = st.text_input('Enter Stock Ticker', 'BCH-USD')
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import datetime as dt
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        if(infoType == 'Fundamental'):
            st.header('Not Available For Crypto Currency. Check Technical Analysis')

        else:
            def calcMovingAverage(data, size):
                df = data.copy()
                df['sma'] = df['Adj Close'].rolling(size).mean()
                df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
                df.dropna(inplace=True)
                return df
            def calc_macd(data):
                df = data.copy()
                df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
                df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
                df['macd'] = df['ema12'] - df['ema26']
                df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
                df.dropna(inplace=True)
                return df
            def calcBollinger(data, size):
                df = data.copy()
                df["sma"] = df['Adj Close'].rolling(size).mean()
                df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
                df["width"] = df["bolu"] - df["bold"]
                df.dropna(inplace=True)
                return df
            st.title('Technical Indicators')
            st.subheader('Moving Average')
           

            start = dt.datetime.today()-dt.timedelta(5 * 365)
            end = dt.datetime.today()
            dataMA = yf.download(user_input,start,end)
            df_ma = calcMovingAverage(dataMA, 20)
            df_ma = df_ma.reset_index()
            figMA = go.Figure()
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['Adj Close'],
                        name = "Prices Over Last " + str(5) + " Year(s)"
                    )
                )    
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.add_trace(
                    go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                    )
                )
            figMA.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            figMA.update_layout(legend_title_text='Trend')
            figMA.update_yaxes(tickprefix="inr")
            st.plotly_chart(figMA, use_container_width=True)


            st.subheader('Moving Average Convergence Divergence (MACD)')
            
    
            startMACD = dt.datetime.today()-dt.timedelta(4 * 365)
            endMACD = dt.datetime.today()
            dataMACD = yf.download(user_input,startMACD,endMACD)
            df_macd = calc_macd(dataMACD)
            df_macd = df_macd.reset_index()
    
            figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['Adj Close'],
                        name = "Prices Over Last " + str(4) + " Year(s)"
                    ),
                row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema12'],
                        name = "EMA 12 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['ema26'],
                        name = "EMA 26 Over Last " + str(4) + " Year(s)"
                ),
            row=1, col=1
            )
    
            figMACD.add_trace(
                    go.Scatter(
                        x = df_macd['Date'],
                        y = df_macd['macd'],
                        name = "MACD Line"
                ),
            row=2, col=1
            )
    
            figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
            )
    
            figMACD.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                ))
    
            figMACD.update_yaxes(tickprefix="inrs")
            st.plotly_chart(figMACD, use_container_width=True)

            st.subheader('Bollinger Band')

            startBoll= dt.datetime.today()-dt.timedelta(5 * 365)
            endBoll = dt.datetime.today()
            dataBoll = yf.download(user_input,startBoll,endBoll)
            df_boll = calcBollinger(dataBoll, 20)
            df_boll = df_boll.reset_index()
            figBoll = go.Figure()
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bolu'],
                            name = "Upper Band"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['sma'],
                            name = "SMA" + str(20) + " Over Last " + str(5) + " Year(s)"
                        )
                )
    
    
            figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bold'],
                            name = "Lower Band"
                        )
                )
    
            figBoll.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0
            ))
    
            figBoll.update_yaxes(tickprefix="$")
            st.plotly_chart(figBoll, use_container_width=True)


            
        

        
        




    
if selected_status == 'About':
    st.title('About')
    st.markdown('''
    # Know everything about trading:
    ## What is trading?
    #### Trading is one of the most profitable activities on the Internet. Every day, billions are earned and lost at the click of a mouse by people who trade online. As a trader, you trade in financial instruments such as stocks, currency pairs, and index funds. But why is trading so advantageous now, how do you do it, and what are the characteristics of trading?

    ## How can you trade yourself?
    #### Do you want to trade yourself? Then you need a free account with a broker. A broker is a party who, for example, makes it possible to buy and sell shares quickly. With most brokers, you can try online trading completely without risk by using a demo.

    ## What is trading?
    #### Trading is a specific way of investing. You actively trade in, for example, a share. Trading differs from ‘traditional investing’:
    + ### Investing: buying a share to achieve price gains in the long term.
    + ### Trading: buying and selling a stock quickly.

    ## How can you make money trading?
    #### Making money from trading is certainly possible. As you may know, the share prices, for example, are moving up and down every day. This movement is also called volatility  and as a trader, you can benefit from these movements. You can place orders on rising and falling prices. The latter we also call going short. When you go short you predict that the price will fall, and you get a positive result when this eventually happens.
    #### Here are two broad strategies to choose from: intraday trading and day trading. When day trading,  you keep an eye on the price developments and open multiple positions in one day. These positions can be kept open for a longer period. For people with less time,  intraday trading is more attractive. When you do intraday trading, you open and close positions on the same day.

    ## Learning to trade: become a good trader
    #### Nowadays, anyone can trade online. Yet, very few people manage to become successful. If you want to become a good trader, you will have to apply a strategy. On the Internet you can find strategies that can help you to achieve better results.

    ## In which securities can you trade?
    #### When you start trading, you can trade in different securities. Below is a small list of the different securities in which you can trade:
    + ### Stock Market
    + ### Crypto Currencies
    #### The basics of trading are simple: you buy when you expect the price to go up, and you sell when you expect it to decrease! However, the implementation is not necessarily simple. In online trading,  emotions play an important role. Many people are afraid of losing and therefore keep their loss open for too long. In the meantime, they are also afraid to lose their winnings, which means that they close the position at a small profit.
    #### In the end, it is smart to do the opposite when you are trading. It is better to cut losses and to allow profits to continue. Therefore, always ensure a favourable ratio between your  risk  and return. 

    ## What is the best way to practice?
    #### You must practice a lot! Trading is a skill  you do not learn from books. Of course, you can learn aspects of trading, such as technical and fundamental analysis, from a book. However, only theoretical knowledge is not enough. Therefore, it is important to practice often. Try out different investment methods and see if you manage to achieve a positive result.

    ## How do you become a good trader?
    #### After you have read this article, you will know exactly how to get started with online trading. But of course, just trading online is not enough: You also want to achieve good results. In this part of the article we look at what you need to consider when you want to make money from trading.
    ### Follow the trend
    #### In many industries, it works well to be opinionated. Artists make more money when they create something unique and as an entrepreneur, you better come up with an original plan. Unfortunately, when you start trading, this works a little differently!
    #### If you want to get a good result, it is wise to trade with the trend as much as possible.  Creativity is therefore often not beneficial. Investors sometimes say the trend is your friend. By buying when prices are mainly rising and selling when prices are mainly falling, you greatly increase the chances of success.
    ### Take a break on occasion
    #### Similarly, boredom is not the trader’s enemy. It is better not to have a position at all than to lose a lot of money. When it is unclear where the market is going, it is better not to act.Even when there is a lot of uncertainty in the market, sometimes it is better to wait and see.
    ### Have a plan
    #### The best traders understand that they need a plan to deliver good results. First, determine the amount of money you want to trade with. This is your ‘corporate capital’. Based on this, you can determine the size of the positions you can take.
    ### Everyone seems to want to have an opinion about everything. Racial matters, the climate or the future of the European Union. Everyone is an expert these days. As a trader, it is better not to have an opinion.
    ### Do not act based on what you think is good or bad. The stock markets often anticipate good news or bad news. When good news is expected, people start buying shares. As soon as the news is presented, you see that many traders take their profits, which can cause the price to fall. So, it is not about your opinion, it is about the actions of large groups of traders.
    
    ### Set simple rules
    #### Finally, it is important to draw up simple rules. The most complex system is often not the best trading system. A good trader needs a lot of discipline. When you make the rules too complicated, it is a lot easier to (accidentally) deviate from them.
    #### Therefore, set some rules that determine whether you open a trade or not. Then evaluate these rules constantly and adjust where necessary.

    ### Trading as a profession
    #### Many people encounter trading while they hold another job. They trade in addition to their profession. However, it is also possible to trade professionally within a company.

    ### What is a trader?
    #### A trader is someone who actively trades on the market. Anyone can become a trader. To do so, you only need to open an account with an online broker.

    ### What is day trading?
    #### Day trading also means within the day. Many securities are traded within a session. For example, shares are traded during the opening hours of the stock exchange. Day traders try to take advantage of these fluctuations by taking one or more positions during this session. A Day trader will close his positions before the end of the trading session.

    ### Is there such a thing as rapid trading?
    #### In general, rapid trading exists. Traders often open multiple positions in one day. However, you can also trade long-term. This is what we call day intraday trading. In intraday trading, positions can sometimes be held for a few days to weeks. However, this is a lot faster than traditional investing. In traditional investments,  shares are often held for many years.

    ### Can you trade as a beginner?
    #### Beginners can also start trading. Many brokers offer the option to try out the possibilities completely without risk with a demo. This allows you to make some serious mistakes before you make your first deposit. It is advisable to start with a small amount of money

    ### What is online trading?
    #### All trading nowadays is done online. In the past, you had to call the bank to buy or sell shares. Fortunately, this changed. All you have to do is log into your account and you can quickly buy and sell shares.




    ## Understanding Stock Market Analysis
    #### Stock market analysis can be divided into two parts- Fundamental Analysis and Technical Analysis.

    ### 1. Fundamental Analysis-
    #### This includes analyzing the current business environment and finances to predict the future profitability of the company.

    ### 2. Technical Analysis-
    #### This deals with charts and statistics to identify trends in the stock market.

    #### There are so many factors involved in the prediction of stock market performance hence it becomes one of the most difficult things to do especially when high accuracy is required.

    ## What is ticker?
    #### As you watched a financial network or checked out a market web site knows, security prices, particularly those of stocks, are frequently on the move. A stock ticker is a report of the price of certain securities, updated continuously throughout the trading session by the various stock market exchanges.
    #### A "tick" is any change in the price of the security, whether that movement is up or down. A stock ticker automatically displays these ticks, along with other relevant information, like trading volume, that investors and traders use to stay informed about current market conditions and the interest in that particular security.

    ## Where you get tickers?
    #### You can get tickers in various market web sites, but i suggest yahoo finance.
    #### click here to get tickers https://in.finance.yahoo.com/

    ### What is an uptrend?
    #### An uptrend is a bullish trend in the market. Prices will continuously increase over a certain period of time. They create higher peaks after peaks and higher troughs after troughs.
    #### in our model in moving average graph when 100 days moving average crosses above 365 days moving average then these is a point of starting uptrend.

    ### What is a downtrend?
    #### A downtrend describes the movement of a stock towards a lower price from its previous state. It will exist as long as there is a continuation of lower highs and lower lows in the stock chart. The downtrend is reversed once the conditions are no longer met.
    #### in our model in moving avearge graph when 100 days moving avaerage below 365 days moving average then these is a point of starting downtrend.

    ## How to use?
    #### First open the sidebar then select trading option which you want.
    #### Then select ticker if you don't know what is ticker then check above article, then selected ticker write in select ticker section and hit the enter button.
    #### how to read graph or how to predict? then you have to know what is uptrend and downtrend if you don't know then we mentionend in above article, and how it's useful in our model that is also mention please check.

    ## What is MACD?
    #### Moving average convergence divergence (MACD) is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.

    ## What is Moving Average?
    #### A moving average is a technical indicator that investors and traders use to determine the trend direction of securities.

    ## What is Bollinger Band?
    #### These Bands depict the volatility of stock as it increases or decreases. The bands are placed above and below the moving average line of the stocks. The wider the gap between the bands, higher is the degree of volatility.
    ''')

   


   

   





    





st.subheader('NOTE: These Are Only Predictions We Cannot Give Any Guarentee,Invest At Your Own Risk.')
st.subheader('Happy Investing...')













