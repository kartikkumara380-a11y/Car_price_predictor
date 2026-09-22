# Car_price_predictor
Web application with machine learning model built using Python for used car's resale value based on car attributes. Developed with Streamlit & A Trained Regression Model ( Loaded as joblib ) Having the R Score Around 86% Accuracy.

Features:
-Recives Input for the User:Brand, Current Price of the Car, Car Age in Years, No. of KMS driven, Fuel Type, Transmission type, Type of Seller (dealership or private), Previous owners number.
-To optimize prediction, engineers select features such as Price per Age and Price per Km.
-One-hot encodes categorical variables (brand, fuel type, seller type, transmission) to conform to the input shape that the model was expecting
-Indicates the predicted resale price and an estimate of the range of values (8%)
-Display a bar chart for original price Vs. resale value estimate
-Calculates and shows the percentage depreciation from the initial price
-Styled UI - hero section, styled buttons, and gradient result card
Tech stack: Python, Streamlit, pandas, scikit-learn (model), joblib (model persistence)
