import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Load the dataset
df = pd.read_csv("IPL_2023_Matches.csv")


# 1. Data Cleaning  
# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna({
    'player_of_the_match': 'Unknown',
    'venue': 'Unknown',
    'winner': 'No Result',
    'result_margin': 0,
}, inplace=True)

# Ensure correct dtypes
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 2. Feature Engineering  
# New feature: Is a close match (margin <= 10 runs or <= 5 wickets)
df['close_match'] = ((df['result_margin'] <= 10) & (df['result'] == 'runs')) | \
                    ((df['result_margin'] <= 5) & (df['result'] == 'wickets'))

# Extract month
df['month'] = df['date'].dt.month_name()

# Match played in evening or day (assuming time column available)
# df['match_time'] = pd.to_datetime(df['time'], errors='coerce')
# df['match_session'] = df['match_time'].dt.hour.apply(lambda x: 'Evening' if x >= 16 else 'Day')

# 3. Data Integrity Check  
# Check for invalid team names
valid_teams = df['team1'].unique().tolist() + df['team2'].unique().tolist()
if not set(df['winner'].unique()).issubset(set(valid_teams + ['No Result'])):
    print("Warning: Winner column has unknown teams.")

# 4. Summary Statistics

print("Basic Description:\n", df.describe(include='all'))
print("\nTop Winning Teams:\n", df['winner'].value_counts().head(5))

# 5. Outliers & Anomalies
# Plot to check margin outliers
plt.figure(figsize=(10, 4))
sns.boxplot(x='result', y='result_margin', data=df)
plt.title('Outlier Detection in Match Margins')
plt.show()

# 6. Visual Patterns
# Win count per team
plt.figure(figsize=(10, 5))
sns.countplot(y='winner', data=df, order=df['winner'].value_counts().index)
plt.title("Number of Wins per Team")
plt.xlabel("Win Count")
plt.ylabel("Team")
plt.show()

# Close matches by month
plt.figure(figsize=(10, 5))
sns.countplot(x='month', hue='close_match', data=df)
plt.title("Close Matches Distribution by Month")
plt.show()

# 7. Natural Language Query using Pandas AI
# Setup Pandas AI
llm = OpenAI(api_token="your-openai-api-key")  # Replace with your key
df_smart = SmartDataframe(df, config={"llm": llm})

# Example queries
response1 = df_smart.chat("Which team won the most matches in IPL 2023?")
print("AI Insight 1:", response1)

response2 = df_smart.chat("Tell me which month had the most close matches?")
print("AI Insight 2:", response2)

response3 = df_smart.chat("Are there any anomalies in match margins?")
print("AI Insight 3:", response3)
