import streamlit as st
import pandas as pd
from sklearn.model_selection import LinearRegression
df = pd.read_csv("data.csv")
X = df.iloc[:, :-1].values  # features
Y = df.iloc[:, -1].values
X = df[['hours studied']]
y = df['examscore']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,state=42)
model=LinearRegression()
model.fit(X_train,y_train)
st.title("Exam score predictor")
st.write("Enter hours studeied")
hours= st.number_input("hours studied",min_value=0.0, step=0.1)
if st.button("predict score"):
  predicted_score= model.predict([[hours]])[0]
  st.success(f"predicted score:{predicted_score:.2f})
  st.write("sample training data")
  st.dataframe(df)
