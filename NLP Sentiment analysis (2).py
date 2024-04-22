#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Collection 

import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\shreya\\OneDrive\\Desktop\\AamAadmiParty.csv')


# In[2]:


print(df)


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


#Checking null values
print(df.isnull().sum())
print("No. of rows:",len(df.axes[0]))


# In[8]:


#Checking duplicate rows
print("No. of Duplicated Rows:", df.duplicated().sum())


# In[9]:


import pandas as pd

# Assuming 'df' is your DataFrame
# Drop rows with null values in specified columns
columns_to_check = ['Tweet Id', 'Text', 'Username', 'likeCount']
df_cleaned = df.dropna(subset=columns_to_check)

# Print the number of rows after removing null values
print("No. of rows after removing null values in specified columns:", len(df_cleaned.axes[0]))


# In[10]:


# Check for null values after removing
print("Null values after removing:")
print(df_cleaned.isnull().sum())


# In[11]:


missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)


# In[12]:


import pandas as pd

# Assuming 'df' is your DataFrame
# Drop rows with missing values
df_cleaned = df.dropna()

# Print the number of rows after removing missing values
print("No. of rows after removing missing values:", len(df_cleaned))


# In[13]:


# Check for missing values in cleaned DataFrame
missing_values_count = df_cleaned.isnull().sum()

# Print the count of missing values in each column
print("Missing values in the cleaned DataFrame:")
print(missing_values_count)


# In[14]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Assuming 'df' is your DataFrame with columns 'Datetime', 'Tweet Id', 'Text', 'Username', 'likeCount'

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to 'Username'
df['Username_encoded'] = label_encoder.fit_transform(df['Username'])

# Print the DataFrame with encoded 'Username'
print(df)


# In[15]:


df


# In[16]:


label_encoder = LabelEncoder()
categorical_columns = ['Text', 'Username']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])


# In[17]:


# Save the dataset to a CSV file
df.to_csv('AamAdmiPartyEncoded.csv', index=False) 


# In[18]:


df


# In[19]:


#Checking null values
print(df.isnull().sum())
print("No. of rows:",len(df.axes[0]))


# In[20]:


missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)


# In[21]:


# Data Preprocessing 


# In[22]:


# Preprocessing
import re  # Import the 're' module for regular expressions
from nltk.tokenize import word_tokenize  # Import word_tokenize from NLTK for tokenization
import pandas as pd


def clean_text(text):
    if isinstance(text, str):  # Check if text is a string
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return cleaned_text.lower()
    else:
        return ''  # Return empty string for non-string values

df['cleaned_text'] = df['Text'].apply(clean_text)
df['tokens'] = df['cleaned_text'].apply(word_tokenize)
print(df['cleaned_text'])


# In[23]:


print(df['cleaned_text'])


# In[24]:


print(df['tokens'])


# In[25]:


import nltk
nltk.download('punkt')


# In[26]:


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sentiment Analysis using VADER
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])


# In[27]:


print(df[['cleaned_text', 'sentiment_score']])


# In[28]:


import nltk
nltk.download('vader_lexicon')


# In[29]:


features = df[['Datetime', 'Tweet Id', 'Text', 'Username', 'Username_encoded']]
target = df['likeCount']


# In[30]:


df.columns = df.columns.tolist()
print(df.columns)


# In[31]:


df


# In[32]:


data=df.to_csv('NLPSENTIMENT.csv', index=False) 


# In[33]:


from sklearn.model_selection import train_test_split

#Step 5: Splitting Data into Train and Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[34]:


print(pd.DataFrame(X_train).head())


# In[35]:


X.head()


# In[36]:


print(pd.DataFrame(y_train).head())


# In[37]:


y.head()


# In[38]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define columns to scale (excluding non-numerical columns)
numerical_columns = ['likeCount']  # Add other numerical columns if needed

# Extract features and target variable
features = df.drop(columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'Username_encoded'])
target = df['likeCount']

# Apply Min-Max scaling to numerical features
scaler = MinMaxScaler()
features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

# Display the scaled features DataFrame
print(features)


# In[39]:


features


# In[40]:


import pandas as pd

correlation_matrix = df.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[41]:


import pandas as pd

# Assuming 'df' is your DataFrame containing the dataset

# Define a function to handle unique values for different data types
def uniquevals(col):
    if isinstance(df[col].iloc[0], list):  # Check if the first element is a list
        unique_values = set()
        for sublist in df[col]:
            unique_values.update(sublist)  # Add elements from each list to the set
        print(f'Details of the particular col {col} is : {list(unique_values)}')
    else:
        print(f'Details of the particular col {col} is : {df[col].unique()}')

# Loop through columns and print unique values
for col in df.columns:
    uniquevals(col)
    print("-" * 75)


# In[42]:


# viewing the distribution of the InsuranceCost column
import seaborn as sns
sns.distplot(df['likeCount'],color='red')


# In[43]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Plot the boxplot for 'sentiment_score' column
plt.figure(figsize=(5, 6))
sns.boxplot(y=df['sentiment_score'])
plt.show()


# In[44]:


numerical_var = list(df.describe().columns[1:])
numerical_var


# In[45]:


# Count plot for the categorical features

for col in numerical_var:
  plt.figure(figsize=(10,12))
  sns.countplot(data=df,x=col)
  plt.title(col)
  plt.show()


# In[46]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Assuming 'data_encoded' is your DataFrame and 'numerical_var' is a list of numerical column names
for col in numerical_var:
    # Check for NaN values and drop rows containing NaN
    df.dropna(subset=[col, 'likeCount'], inplace=True)
    
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df[[col]])
    y = df['likeCount']
    
    # Fit a Ridge regression model with regularization parameter alpha
    model = Ridge(alpha=1.0)  # Adjust alpha as needed
    model.fit(X, y)
    
    # Plotting
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    plt.scatter(X, y)
    plt.xlabel(col)
    plt.ylabel('likeCount')
    ax.set_title('likeCount vs ' + col)
    
    # Plot the regression line
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    
plt.show()


# In[47]:


import matplotlib.pyplot as plt

# Check for and handle missing values in 'Username' column
df.dropna(subset=['Datetime','likeCount'], inplace=True)

# List of categorical variables you want to create pie charts for
categorical_variables = ['Datetime','likeCount']

# Set up subplots for the pie charts
fig, axes = plt.subplots(1, len(categorical_variables), figsize=(15, 7))

# Ensure axes is iterable even for a single subplot
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Iterate through each categorical variable and create a pie chart
for i, categorical_variable in enumerate(categorical_variables):
    category_counts = df[categorical_variable].value_counts()
    
    # Limit the number of categories displayed in the pie chart for clarity
    max_categories = 10  # Adjust as needed
    if len(category_counts) > max_categories:
        category_counts = category_counts[:max_categories]
    
    axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Distribution of {categorical_variable}')

plt.tight_layout()
plt.show()


# In[48]:


import matplotlib.pyplot as plt

# Check for and handle missing values in 'Username' column
df.dropna(subset=['Tweet Id','Username'], inplace=True)

# List of categorical variables you want to create pie charts for
categorical_variables = ['Tweet Id','Username']

# Set up subplots for the pie charts
fig, axes = plt.subplots(1, len(categorical_variables), figsize=(15, 7))

# Ensure axes is iterable even for a single subplot
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Iterate through each categorical variable and create a pie chart
for i, categorical_variable in enumerate(categorical_variables):
    category_counts = df[categorical_variable].value_counts()
    
    # Limit the number of categories displayed in the pie chart for clarity
    max_categories = 10  # Adjust as needed
    if len(category_counts) > max_categories:
        category_counts = category_counts[:max_categories]
    
    axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Distribution of {categorical_variable}')

plt.tight_layout()
plt.show()


# In[49]:


import matplotlib.pyplot as plt

# Check for and handle missing values in 'Username' column
df.dropna(subset=['sentiment_score','Text'], inplace=True)

# List of categorical variables you want to create pie charts for
categorical_variables = ['sentiment_score','Text']

# Set up subplots for the pie charts
fig, axes = plt.subplots(1, len(categorical_variables), figsize=(15, 7))

# Ensure axes is iterable even for a single subplot
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Iterate through each categorical variable and create a pie chart
for i, categorical_variable in enumerate(categorical_variables):
    category_counts = df[categorical_variable].value_counts()
    
    # Limit the number of categories displayed in the pie chart for clarity
    max_categories = 10  # Adjust as needed
    if len(category_counts) > max_categories:
        category_counts = category_counts[:max_categories]
    
    axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Distribution of {categorical_variable}')

plt.tight_layout()
plt.show()


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define features and target variable
features = df[['Datetime', 'Tweet Id', 'Text', 'Username', 'Username_encoded']]
target = df['likeCount']

# Combine features and target into a single DataFrame
data = pd.concat([features, target], axis=1)

# Plot pairplot for all variables
sns.pairplot(data)
plt.suptitle('Pairplot of Features and Target (likeCount)', y=1.02)
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame containing the data
features = df[['Datetime', 'likeCount']]

# Plot histogram for 'likeCount'
plt.figure(figsize=(8, 6))
plt.hist(features['likeCount'], bins=10)  # Adjust the number of bins as needed
plt.xlabel('Like Count')
plt.ylabel('Frequency')
plt.title('Histogram of Like Counts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot histogram for 'Datetime'
plt.figure(figsize=(8, 6))
plt.hist(features['Datetime'], bins=10)  # Adjust the number of bins as needed
plt.xlabel('Datetime')
plt.ylabel('Frequency')
plt.title('Histogram of Datetime')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[1]:


import pandas as pd

df = pd.read_csv('C:\\Users\\shreya\\OneDrive\\Desktop\\AamAadmiParty.csv')

# Define your features (independent variables) and the target variable
features = df[['Datetime', 'Tweet Id', 'Text', 'Username']]
target = df['likeCount']


# Calculate the correlation coefficients correctly using pd.concat()
correlations = pd.concat([features, target], axis=1).corr().abs()

# Display the correlation coefficients
print(correlations['likeCount'])

#Positive values indicate a positive correlation, negative values indicate a negative correlation, 
#and values close to 0 suggest a weak correlation


# In[2]:


X = df.drop(columns='likeCount',axis=1)
Y = df['likeCount']


# In[3]:


print(X)


# In[4]:


print(Y)


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame df with columns 'Datetime', 'Tweet Id', 'Text', 'Username', and 'likeCount'
features = ['Datetime', 'Tweet Id', 'Text', 'Username']
target = 'likeCount'

X = df[features]  # Features or inputs
y = df[target]  # Target variable or labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


print(df.columns)  # Print the column names of your DataFrame


# In[7]:


print(X)


# In[8]:


print(Y)


# In[9]:


print(X_train)


# In[10]:


print(y_train)


# In[11]:


print(X_test)


# In[12]:


print(y_test)


# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame df with columns 'Datetime', 'Tweet Id', 'Text', 'Username', and 'likeCount'
# Preprocess 'Datetime' column to extract relevant features, handling errors
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df['year'] = df['Datetime'].dt.year
df['month'] = df['Datetime'].dt.month
df['day'] = df['Datetime'].dt.day
df['hour'] = df['Datetime'].dt.hour
df['minute'] = df['Datetime'].dt.minute

# Define features and target
features = ['year', 'month', 'day', 'hour', 'minute', 'Tweet Id']  # Adjust features as needed
target = 'likeCount'

# Drop rows with NaN values in the target variable 'likeCount'
df = df.dropna(subset=[target])

X = df[features]  # Features or inputs
y = df[target]  # Target variable or labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can train your linear regression model and evaluate it as before
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")


# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assuming you have a DataFrame df with columns 'year', 'month', 'day', 'hour', 'minute', 'Tweet Id', and 'likeCount'
# Preprocess 'Datetime' column to extract relevant features, handling errors
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df['year'] = df['Datetime'].dt.year
df['month'] = df['Datetime'].dt.month
df['day'] = df['Datetime'].dt.day
df['hour'] = df['Datetime'].dt.hour
df['minute'] = df['Datetime'].dt.minute

# Define features and target
features = ['year', 'month', 'day', 'hour', 'minute', 'Tweet Id']  # Adjust features as needed
target = 'likeCount'

# Drop rows with NaN values in the target variable 'likeCount'
df = df.dropna(subset=[target])

X = df[features]  # Features or inputs
y = df[target]  # Target variable or labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get predicted values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot actual vs predicted values for the training set
plt.scatter(y_train, y_pred_train, color='blue', label='Actual vs Predicted (Training)')
plt.xlabel('Actual Likes')
plt.ylabel('Predicted Likes')
plt.title('Actual vs Predicted Likes (Training Set)')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs predicted values for the test set
plt.scatter(y_test, y_pred_test, color='green', label='Actual vs Predicted (Test)')
plt.xlabel('Actual Likes')
plt.ylabel('Predicted Likes')
plt.title('Actual vs Predicted Likes (Test Set)')
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('NLPSENTIMENT.csv')

# Check unique classes in the 'sentiment_score' column
unique_classes = df['sentiment_score'].unique()

# Print the unique classes
print(unique_classes)


# In[16]:


import pandas as pd

# Load your dataset (assuming it's stored in a DataFrame named df)
df = pd.read_csv('C:\\Users\\shreya\\OneDrive\\Desktop\\AamAadmiParty.csv')

# Define sentiment thresholds
threshold_positive = 500  # Adjust this threshold as needed
threshold_negative = 100  # Adjust this threshold as needed

# Function to categorize sentiment based on likeCount
def categorize_sentiment(like_count):
    if like_count > threshold_positive:
        return 'Positive'
    elif like_count < threshold_negative:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis and create 'Sentiment' column
df['Sentiment'] = df['likeCount'].apply(categorize_sentiment)

# Save the updated DataFrame to a CSV file
df.to_csv('C:\\Users\\shreya\\OneDrive\\Desktop\\AamAadmiParty_with_sentiment.csv', index=False)

# Display the updated DataFrame with sentiment labels
print(df)


# In[17]:


df


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame (replace this with your actual DataFrame)
# df = pd.read_csv('your_dataset.csv')

# Assuming 'Sentiment' column contains 'Positive', 'Negative', 'Neutral' labels
sentiment_counts = df['Sentiment'].value_counts()

# Plotting the bar graph
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Analysis Results')
plt.xticks(rotation=0)
plt.tight_layout()

# Display the plot
plt.show()


# The sentiment analysis results based on political party-related tweets are displayed in this bar graph. The x-axis shows the sentiment categories: 'Negative', 'Neutral', and 'Positive'. The amount of tweets classified into each sentiment is shown on the y-axis.
# 
# The quantity of tweets with a "Negative" emotion classification is shown by the green bar. It suggests that a sizable portion of tweets about the political party are critical.
# The quantity of tweets with a "Neutral" emotion classification is shown by the red bar. There are a lot of neutral tweets displayed, indicating a balance between positive and negative feelings.
# The quantity of tweets categorized as "Positive" emotion is shown by the blue bar. It suggests that the proportion of positive tweets to negative and neutral ones is lower.
# In general, this graph

# In[187]:


import pandas as pd

# Sample DataFrame (replace this with your actual DataFrame)
# df = pd.read_csv('your_dataset.csv')

# Assuming 'Sentiment' column contains 'Positive', 'Negative', 'Neutral' labels
sentiment_counts = df['Sentiment'].value_counts()

# Calculate total number of tweets
total_tweets = sentiment_counts.sum()

# Calculate percentage of each sentiment category
percentage_positive = (sentiment_counts['Positive'] / total_tweets) * 100
percentage_neutral = (sentiment_counts['Neutral'] / total_tweets) * 100
percentage_negative = (sentiment_counts['Negative'] / total_tweets) * 100

print("Percentage of Positive tweets:", percentage_positive)
print("Percentage of Neutral tweets:", percentage_neutral)
print("Percentage of Negative tweets:", percentage_negative)


# These percentages represent the distribution of sentiments within the dataset of tweets:
# 
# Percentage of Positive tweets: Approximately 21.54% of the tweets in the dataset are classified as having a positive sentiment. This indicates that a relatively smaller portion of the tweets convey positive emotions, opinions, or expressions.
# Percentage of Neutral tweets: Around 37.37% of the tweets are classified as having a neutral sentiment. These tweets likely contain information or statements that do not convey strong positive or negative emotions.
# Percentage of Negative tweets: The majority of the tweets, approximately 41.08%, are classified as having a negative sentiment. This suggests that a significant portion of the tweets express negative emotions, opinions, or sentiments.
# Overall, the analysis reveals that the dataset contains a higher proportion of negative tweets compared to positive ones, indicating a prevalence of negative sentiment among the tweets analyzed.

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('C:\\Users\\shreya\\OneDrive\\Desktop\\AamAadmiParty_with_sentiment.csv')

# Check the unique classes in the 'Sentiment' column
unique_classes = df['Sentiment'].unique()
if len(unique_classes) < 2:
    raise ValueError("The dataset must have at least two classes for classification.")

# Check for NaN values in the 'Text' column
nan_indices = df['Text'].isnull()
if nan_indices.any():
    # Handle NaN values by replacing them with empty strings
    df['Text'].fillna('', inplace=True)

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(df['Text'].astype(str))  # Convert to string
y = df['Sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print(classification_report(y_test, y_pred))

# Count Sentiment Labels
sentiment_counts = pd.Series(y_pred).value_counts()

# Calculate Distribution
total_predictions = len(y_pred)
positive_percentage = sentiment_counts.get('positive', 0) / total_predictions * 100
negative_percentage = sentiment_counts.get('negative', 0) / total_predictions * 100
neutral_percentage = sentiment_counts.get('neutral', 0) / total_predictions * 100

print('Positive Sentiment Percentage:', positive_percentage)
print('Negative Sentiment Percentage:', negative_percentage)
print('Neutral Sentiment Percentage:', neutral_percentage)


# In[20]:


pip install faker


# In[21]:


import pandas as pd
from faker import Faker
import random

# Initialize Faker to generate fake data
fake = Faker()

# Number of records to generate
num_records = 6663

# Generate data
data = {
    'Datetime': [fake.date_time_between(start_date='-1y', end_date='now', tzinfo=None) for _ in range(num_records)],
    'Tweet Id': [fake.random_number(digits=18) for _ in range(num_records)],
    'Text': [fake.paragraph() for _ in range(num_records)],
    'Username': [fake.user_name() for _ in range(num_records)],
    'likeCount': [random.randint(0, 1000) for _ in range(num_records)]
}

# Create DataFrame
df1 = pd.DataFrame(data)

# Save to CSV
df1.to_csv('twitter_records_6663.csv', index=False)


# In[22]:


df1


# In[23]:


df1.info()


# In[24]:


df1.describe()


# In[25]:


df1.shape


# In[26]:


#Checking null values
print(df1.isnull().sum())
print("No. of rows:",len(df.axes[0]))


# In[27]:


#Checking duplicate rows
print("No. of Duplicated Rows:", df1.duplicated().sum())


# In[28]:


import pandas as pd

# Assuming 'df' is your DataFrame
# Drop rows with null values in specified columns
columns_to_check = ['Tweet Id', 'Text', 'Username', 'likeCount']
df1_cleaned = df1.dropna(subset=columns_to_check)

# Print the number of rows after removing null values
print("No. of rows after removing null values in specified columns:", len(df1_cleaned.axes[0]))


# In[29]:


# Check for null values after removing
print("Null values after removing:")
print(df1_cleaned.isnull().sum())


# In[30]:


missing_values = df1.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)


# In[31]:


import pandas as pd

# Assuming 'df' is your DataFrame
# Drop rows with missing values
df1_cleaned = df1.dropna()

# Print the number of rows after removing missing values
print("No. of rows after removing missing values:", len(df1_cleaned))


# In[32]:


# Check for missing values in cleaned DataFrame
missing_values_count = df1_cleaned.isnull().sum()

# Print the count of missing values in each column
print("Missing values in the cleaned DataFrame:")
print(missing_values_count)


# In[39]:


df1


# In[41]:


# Convert 'Text' and 'Username' columns to binary encoding (0 and 1)
df1['Text_encoded'] = df1['Text'].apply(lambda x: 1 if x == 'desired_text' else 0)
df1['Username_encoded'] = df1['Username'].apply(lambda x: 1 if x == 'desired_username' else 0)

# Drop the original 'Text' and 'Username' columns
df1.drop(['Text', 'Username'], axis=1, inplace=True)

# Save to CSV
df1.to_csv('twitter_records_binary_encoded.csv', index=False)


# In[42]:


df1


# In[43]:


#Checking null values
print(df1.isnull().sum())
print("No. of rows:",len(df.axes[0]))


# In[44]:


missing_values = df1.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)


# In[46]:


# Preprocessing
import re  # Import the 're' module for regular expressions
from nltk.tokenize import word_tokenize  # Import word_tokenize from NLTK for tokenization
import pandas as pd


def clean_text(text):
    if isinstance(text, str):  # Check if text is a string
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return cleaned_text.lower()
    else:
        return ''  # Return empty string for non-string values

df1['cleaned_text'] = df1['Text_encoded'].apply(clean_text)
df1['tokens'] = df1['cleaned_text'].apply(word_tokenize)
print(df1['cleaned_text'])


# In[48]:


print(df1['cleaned_text'])


# In[49]:


print(df1['tokens'])


# In[50]:


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sentiment Analysis using VADER
sid = SentimentIntensityAnalyzer()
df1['sentiment_score'] = df1['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])


# In[51]:


print(df1[['cleaned_text', 'sentiment_score']])


# In[52]:


df1


# In[53]:


import nltk
nltk.download('vader_lexicon')


# In[54]:


features = df1[['Datetime', 'Tweet Id', 'Text_encoded', 'cleaned_text', 'Username_encoded','tokens','sentiment_score']]
target = df1['likeCount']


# In[55]:


df1.columns = df1.columns.tolist()
print(df1.columns)


# In[56]:


df1


# In[57]:


data=df1.to_csv('Trial.csv', index=False) 


# In[58]:


from sklearn.model_selection import train_test_split

#Step 5: Splitting Data into Train and Split
X = df1.iloc[:, :-1]
y = df1.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[59]:


print(pd.DataFrame(X_train).head())


# In[60]:


X.head()


# In[61]:


print(pd.DataFrame(y_train).head())


# In[62]:


y.head()


# In[65]:


import pandas as pd

correlation_matrix = df1.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[67]:


import pandas as pd

# Assuming 'df1' is your DataFrame containing the dataset

# Define a function to handle unique values for different data types
def uniquevals(col):
    if isinstance(df1[col].iloc[0], list):  # Check if the first element is a list
        unique_values = set()
        for sublist in df1[col]:
            unique_values.update(sublist)  # Add elements from each list to the set
        print(f'Details of the particular col {col} is : {list(unique_values)}')
    else:
        print(f'Details of the particular col {col} is : {df1[col].unique()}')

# Loop through columns and print unique values
for col in df1.columns:
    uniquevals(col)
    print("-" * 75)


# In[68]:


# viewing the distribution of the InsuranceCost column
import seaborn as sns
sns.distplot(df1['likeCount'],color='red')


# In[69]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Plot the boxplot for 'sentiment_score' column
plt.figure(figsize=(5, 6))
sns.boxplot(y=df1['sentiment_score'])
plt.show()


# In[70]:


numerical_var = list(df1.describe().columns[1:])
numerical_var


# In[71]:


# Count plot for the categorical features

for col in numerical_var:
  plt.figure(figsize=(10,12))
  sns.countplot(data=df1,x=col)
  plt.title(col)
  plt.show()


# In[72]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Assuming 'data_encoded' is your DataFrame and 'numerical_var' is a list of numerical column names
for col in numerical_var:
    # Check for NaN values and drop rows containing NaN
    df1.dropna(subset=[col, 'likeCount'], inplace=True)
    
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df1[[col]])
    y = df1['likeCount']
    
    # Fit a Ridge regression model with regularization parameter alpha
    model = Ridge(alpha=1.0)  # Adjust alpha as needed
    model.fit(X, y)
    
    # Plotting
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    plt.scatter(X, y)
    plt.xlabel(col)
    plt.ylabel('likeCount')
    ax.set_title('likeCount vs ' + col)
    
    # Plot the regression line
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    
plt.show()


# In[73]:


import matplotlib.pyplot as plt

# Check for and handle missing values in 'Username' column
df.dropna(subset=['Datetime','likeCount'], inplace=True)

# List of categorical variables you want to create pie charts for
categorical_variables = ['Datetime','likeCount']

# Set up subplots for the pie charts
fig, axes = plt.subplots(1, len(categorical_variables), figsize=(15, 7))

# Ensure axes is iterable even for a single subplot
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Iterate through each categorical variable and create a pie chart
for i, categorical_variable in enumerate(categorical_variables):
    category_counts = df1[categorical_variable].value_counts()
    
    # Limit the number of categories displayed in the pie chart for clarity
    max_categories = 10  # Adjust as needed
    if len(category_counts) > max_categories:
        category_counts = category_counts[:max_categories]
    
    axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Distribution of {categorical_variable}')

plt.tight_layout()
plt.show()


# In[77]:


import matplotlib.pyplot as plt

# Check for and handle missing values in 'Username' column
df.dropna(subset=['Tweet Id'], inplace=True)

# List of categorical variables you want to create pie charts for
categorical_variables = ['Tweet Id']

# Set up subplots for the pie charts
fig, axes = plt.subplots(1, len(categorical_variables), figsize=(15, 7))

# Ensure axes is iterable even for a single subplot
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Iterate through each categorical variable and create a pie chart
for i, categorical_variable in enumerate(categorical_variables):
    category_counts = df1[categorical_variable].value_counts()
    
    # Limit the number of categories displayed in the pie chart for clarity
    max_categories = 10  # Adjust as needed
    if len(category_counts) > max_categories:
        category_counts = category_counts[:max_categories]
    
    axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Distribution of {categorical_variable}')

plt.tight_layout()
plt.show()


# In[78]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define features and target variable
features = df1[['Datetime', 'Tweet Id', 'Text_encoded', 'cleaned_text', 'Username_encoded','tokens','sentiment_score']]
target = df1['likeCount']

# Combine features and target into a single DataFrame
data = pd.concat([features, target], axis=1)

# Plot pairplot for all variables
sns.pairplot(data)
plt.suptitle('Pairplot of Features and Target (likeCount)', y=1.02)
plt.show()


# In[79]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame containing the data
features = df1[['Datetime', 'likeCount']]

# Plot histogram for 'likeCount'
plt.figure(figsize=(8, 6))
plt.hist(features['likeCount'], bins=10)  # Adjust the number of bins as needed
plt.xlabel('Like Count')
plt.ylabel('Frequency')
plt.title('Histogram of Like Counts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot histogram for 'Datetime'
plt.figure(figsize=(8, 6))
plt.hist(features['Datetime'], bins=10)  # Adjust the number of bins as needed
plt.xlabel('Datetime')
plt.ylabel('Frequency')
plt.title('Histogram of Datetime')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[133]:


import pandas as pd

df1 = pd.read_csv('twitter_records_binary_encoded.csv')

# Define your features (independent variables) and the target variable
features = df1[['Datetime','Tweet Id','Text_encoded','Username_encoded']]
target = df1['likeCount']

# Calculate the correlation coefficients correctly using pd.concat()
correlations = pd.concat([features, target], axis=1).corr().abs()

# Display the correlation coefficients
print(correlations['likeCount'])

#Positive values indicate a positive correlation, negative values indicate a negative correlation, 
#and values close to 0 suggest a weak correlation


# In[134]:


X = df1.drop(columns='likeCount',axis=1)
Y = df1['likeCount']


# In[135]:


print(X)


# In[136]:


print(Y)


# In[152]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame df with columns 'Datetime', 'Tweet Id', 'Text', 'Username', and 'likeCount'
features = ['Datetime','Tweet Id','Text_encoded','Username_encoded']
target = 'likeCount'

X = df1[features]  # Features or inputs
y = df1[target]  # Target variable or labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[153]:


df1


# In[154]:


print(df1.columns)  # Print the column names of your DataFrame


# In[155]:


print(X)


# In[156]:


print(Y)


# In[157]:


print(X_train)


# In[158]:


print(y_train)


# In[159]:


print(X_test)


# In[160]:


print(y_test)


# In[161]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame df with columns 'Datetime', 'Tweet Id', 'Text', 'Username', 'likeCount', 'Text_encoded', and 'Username_encoded'
# Preprocess 'Datetime' column to extract relevant features, handling errors
df1['Datetime'] = pd.to_datetime(df1['Datetime'], errors='coerce')
df1['year'] = df1['Datetime'].dt.year
df1['month'] = df1['Datetime'].dt.month
df1['day'] = df1['Datetime'].dt.day
df1['hour'] = df1['Datetime'].dt.hour
df1['minute'] = df1['Datetime'].dt.minute

# Define features and target
features = ['Tweet Id', 'Text_encoded', 'Username_encoded', 'year', 'month', 'day', 'hour', 'minute']
target = 'likeCount'

# Drop rows with NaN values in the target variable 'likeCount'
df1 = df1.dropna(subset=[target])

X = df1[features]  # Features or inputs
y = df1[target]  # Target variable or labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can train your linear regression model and evaluate it as before
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")


# In[162]:


import matplotlib.pyplot as plt

# Generate predictions for the test set
y_pred = model.predict(X_test)

# Plot the best fit line including predicted and actual points
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')  # Scatter plot for actual vs. predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Best fit line
plt.xlabel('Actual Like Count')
plt.ylabel('Predicted Like Count')
plt.title('Actual vs. Predicted Like Count (Linear Regression)')
plt.legend()
plt.grid(True)
plt.show()


# In[172]:


import pandas as pd
import re  # Import the 're' module for regular expressions
from nltk.tokenize import word_tokenize  # Import word_tokenize from NLTK for tokenization
from nltk.corpus import stopwords  # Import NLTK's stopwords
from nltk.stem import PorterStemmer  # Import NLTK's PorterStemmer

# Read the CSV file into a DataFrame
df1 = pd.read_csv('twitter_records_binary_encoded.csv')

# Download NLTK resources (only need to run once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if isinstance(text, str):  # Check if text is a string
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return cleaned_text.lower()
    else:
        return ''  # Return empty string for non-string values

df1['cleaned_text'] = df1['Text_encoded'].apply(clean_text)
df1['tokens'] = df1['cleaned_text'].apply(word_tokenize)

# Remove stop words and perform stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def process_tokens(tokens):
    filtered_tokens = [ps.stem(token) for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

df1['processed_tokens'] = df1['tokens'].apply(process_tokens)
print(df1['processed_tokens'])


# In[173]:


print(df1['cleaned_text'])


# In[174]:


print(df1['tokens'])


# In[175]:


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sentiment Analysis using VADER
sid = SentimentIntensityAnalyzer()
df1['sentiment_score'] = df1['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])


# In[176]:


print(df1[['cleaned_text', 'sentiment_score']])


# In[177]:


df1


# In[179]:


import pandas as pd

# Read the CSV file into a DataFrame
df1 = pd.read_csv('Trial.csv')

# Check unique classes in the 'sentiment_score' column
unique_classes = df1['sentiment_score'].unique()

# Print the unique classes
print(unique_classes)


# In[183]:


import pandas as pd

# Load your dataset (assuming it's stored in a DataFrame named df)
df1 = pd.read_csv('Trial.csv')

# Define sentiment thresholds
threshold_positive = 500  # Adjust this threshold as needed
threshold_negative = 100  # Adjust this threshold as needed

# Function to categorize sentiment based on likeCount
def categorize_sentiment(like_count):
    if like_count > threshold_positive:
        return 'Positive'
    elif like_count < threshold_negative:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis and create 'Sentiment' column
df1['Sentiment'] = df['likeCount'].apply(categorize_sentiment)

# Save the updated DataFrame to a CSV file
df1.to_csv('Trial_sentiment.csv', index=False)

# Display the updated DataFrame with sentiment labels
print(df1)


# In[184]:


df1


# In[185]:


import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame (replace this with your actual DataFrame)
# df = pd.read_csv('your_dataset.csv')

# Assuming 'Sentiment' column contains 'Positive', 'Negative', 'Neutral' labels
sentiment_counts = df1['Sentiment'].value_counts()

# Plotting the bar graph
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Analysis Results')
plt.xticks(rotation=0)
plt.tight_layout()

# Display the plot
plt.show()


# The graph represents the results of sentiment analysis on a dataset of tweets. Here's an explanation:
# 
# X-axis: It shows the different sentiment categories identified in the analysis, which are "Positive", "Neutral", and "Negative".
# Y-axis: This axis represents the number of tweets associated with each sentiment category.
# Bars: Each colored bar represents the count of tweets for a specific sentiment category.
# The green bar represents the number of tweets classified as "Positive" sentiment.
# The red bar represents the number of tweets classified as "Neutral" sentiment.
# The blue bar represents the number of tweets classified as "Negative" sentiment.
# From the graph, it's evident that the majority of the tweets are classified as having a "Positive" sentiment, followed by "Neutral" sentiment, and finally, the least number of tweets are classified as having a "Negative" sentiment. This visualization provides an overview of the sentiment distribution within the analyzed tweets.

# In[186]:


import pandas as pd

# Sample DataFrame (replace this with your actual DataFrame)
# df = pd.read_csv('your_dataset.csv')

# Assuming 'Sentiment' column contains 'Positive', 'Negative', 'Neutral' labels
sentiment_counts = df1['Sentiment'].value_counts()

# Calculate total number of tweets
total_tweets = sentiment_counts.sum()

# Calculate percentage of each sentiment category
percentage_positive = (sentiment_counts['Positive'] / total_tweets) * 100
percentage_neutral = (sentiment_counts['Neutral'] / total_tweets) * 100
percentage_negative = (sentiment_counts['Negative'] / total_tweets) * 100

print("Percentage of Positive tweets:", percentage_positive)
print("Percentage of Neutral tweets:", percentage_neutral)
print("Percentage of Negative tweets:", percentage_negative)


# These percentages represent the distribution of sentiments within the dataset of tweets:
# 
# Percentage of Positive tweets: Approximately 57.51% of the tweets in the dataset are classified as having a positive sentiment. This indicates that the majority of the tweets convey positive emotions, opinions, or expressions.
# Percentage of Neutral tweets: Around 39.52% of the tweets are classified as having a neutral sentiment. These tweets likely contain information or statements that do not convey strong positive or negative emotions.
# Percentage of Negative tweets: Only about 2.97% of the tweets are classified as having a negative sentiment. This suggests that a small portion of the tweets express negative emotions, opinions, or sentiments.
# Overall, the analysis reveals that the dataset contains a higher proportion of positive tweets compared to neutral and negative ones, indicating a generally positive sentiment among the tweets analyzed.

# In[194]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming you have a DataFrame df1 with columns 'Datetime', 'Tweet Id', 'likeCount', 'Text_encoded', 'Username_encoded', and 'Sentiment'

# Preprocess 'Datetime' column to extract relevant features, handling errors
df1['Datetime'] = pd.to_datetime(df1['Datetime'], errors='coerce')
df1['year'] = df1['Datetime'].dt.year
df1['month'] = df1['Datetime'].dt.month
df1['day'] = df1['Datetime'].dt.day
df1['hour'] = df1['Datetime'].dt.hour
df1['minute'] = df1['Datetime'].dt.minute

# Define features and target
features = ['Tweet Id', 'Text_encoded', 'Username_encoded', 'year', 'month', 'day', 'hour', 'minute']
target = 'Sentiment'  # Sentiment is your target variable now

# Drop rows with NaN values in the target variable 'Sentiment'
df1 = df1.dropna(subset=[target])

X = df1[features]  # Features or inputs
y = df1[target]  # Target variable or labels

# Convert sentiment labels to numerical values (0, 1, 2 for example)
# This step is necessary for logistic regression
y_numerical = y.map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_numerical, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
logreg_model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
logreg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[ ]:




