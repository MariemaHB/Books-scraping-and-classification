import streamlit as st
import numpy as np
import pandas as pd
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt 

import joblib

from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


data = 'books.xlsx'  

pipe_lr = joblib.load(open('models/books_clf_sgd_30_04_2022.pkl','rb')) 


@st.cache 
def load_data():
    df = pd.read_excel(data, index_col=0)
    return(df)


df = load_data()


def filedownload(df):
    books = df.to_excel('out.xlsx',index=False)
    href = f'<a href="data:file/books" download="Books.xlsx">Download EXCEL File</a>'
    return href




def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data



def books_per_categ_plot():
      fig = plt.figure(figsize=(20,10))
      sns.set_style('darkgrid')
      sns.set_context('notebook',font_scale=1.5)
      sns.countplot(x='Categories',hue="Categories",data=df)
      plt.legend(loc='upper right')
      st.pyplot(fig)


def books_avg_price_per_categ_plot():
    df1 = pd.DataFrame({'Average price': df.groupby('Categories')['Price'].mean()}).reset_index()
    fig = plt.figure(figsize=(20,10))
    Categories = ['travel','mystery','historical-fiction','sequential-art','classics','philosophy']
    sns.barplot(x='Categories', y='Average price',hue='Categories', data=df1)
    st.pyplot(fig)


def nb_five_stars_per_category_plot():
    dff = df.loc[df['Star']=='Five'] 
    fig = plt.figure(figsize=(20,10))
    sns.set_context('notebook',font_scale=1.5)
    sns.countplot(x='Categories',hue='Categories', data=dff)
    st.pyplot(fig)



def predict_category_of_book(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results 



def main():

      st.title('Books Classifier')

      st.markdown("""
This app retrieves a list of books scraped from books.toscrape.
* **Python libraries:** BeautifulSoup, pandas, streamlit, matplotlib, seaborn ..
* **Data source:** [books.toscrape](https://books.toscrape.com/index.html).
""")



      st.header('Display Books')
      st.write('Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
      st.dataframe(df)

      df_to_xlsx = to_excel(df)
      st.download_button(label='📥 Download Excel File',
                                data=df_to_xlsx ,
                                file_name= 'Books.xlsx')


      if st.button('Show Plots'):
            st.header('Number of books per category')
            books_per_categ_plot()

            st.text("")

            st.header('Distribution of avg price per category')
            books_avg_price_per_categ_plot()

            st.text("")

            st.header('Number of 5 stars per category')
            nb_five_stars_per_category_plot()

      st.text("")

      st.header('Predict category of a book')
        
      with st.form('books_clf_form'):
            raw_text = st.text_area('Type Here')
            submit_text = st.form_submit_button(label='Submit')
        
      if submit_text : 
            col1,col2 = st.columns(2) 

            prediction = predict_category_of_book(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success('Original Title')
                st.write(raw_text)

                st.success('Prediction')
                st.write(prediction)
                st.write('Confidence:{}'.format(np.max(probability)))

            with col2:
                st.success('Prediction Probability')
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=sgd.classes_)
                st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['Categories','Probability']

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Categories',y='Probability',color='Categories')
                st.altair_chart(fig,use_container_width=True)




if __name__ == '__main__' : 
    main()