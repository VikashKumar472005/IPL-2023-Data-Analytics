import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

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



#Review 2
# 1. Chart Selection Examples 

# Bar Chart: Wins per Team (Interactive)
fig = px.bar(
    df['winner'].value_counts().reset_index(),
    x='index', y='winner',
    labels={'index': 'Team', 'winner': 'Wins'},
    color='winner',
    color_continuous_scale='Viridis',
    title='Number of Wins per Team (IPL 2023)'
)
fig.update_layout(
    xaxis_title='Team',
    yaxis_title='Number of Wins',
    showlegend=False,
    template='plotly_white'
)
fig.show()

# Pie Chart: Share of Wins
win_counts = df['winner'].value_counts()
fig2 = px.pie(
    values=win_counts.values,
    names=win_counts.index,
    title='Share of Wins by Team',
    color_discrete_sequence=px.colors.sequential.RdBu
)
fig2.update_traces(textinfo='percent+label')
fig2.show()

# Scatter Plot: Margin vs. Date (with Close Matches Highlighted)
fig3 = px.scatter(
    df,
    x='date', y='result_margin',
    color='close_match',
    hover_data=['team1', 'team2', 'winner'],
    title='Match Margin Over Time (Close Matches Highlighted)',
    labels={'result_margin': 'Result Margin', 'date': 'Date'},
    color_discrete_map={True: 'red', False: 'blue'}
)
fig3.update_layout(template='plotly_white')
fig3.show()

# Line Plot: Close Matches per Month
close_by_month = df.groupby('month')['close_match'].sum().reindex([
    'March', 'April', 'May', 'June'], fill_value=0)
fig4 = px.line(
    close_by_month,
    x=close_by_month.index,
    y='close_match',
    title='Number of Close Matches per Month',
    markers=True,
    labels={'close_match': 'Close Matches', 'month': 'Month'}
)
fig4.update_traces(line_color='green')
fig4.show()

# 2. Aesthetics and Clarity

# Example: Improved Bar Chart with Matplotlib
plt.figure(figsize=(10, 6))
sns.barplot(
    y=win_counts.index,
    x=win_counts.values,
    palette='Set2'
)
plt.title('Number of Wins per Team (IPL 2023)', fontsize=16)
plt.xlabel('Wins', fontsize=14)
plt.ylabel('Team', fontsize=14)
for i, v in enumerate(win_counts.values):
    plt.text(v + 0.2, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()

# 3. Interactivity (Plotly) 

# Interactive Filtering: Wins by Team (Dropdown)
teams = df['winner'].unique()
fig5 = px.bar(
    df[df['winner'].isin(teams)],
    x='winner',
    color='winner',
    title='Interactive: Wins by Team',
    labels={'winner': 'Team'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig5.update_layout(
    updatemenus=[
        dict(
            buttons=[
                dict(label=team, method='update',
                     args=[{'x': [[team]], 'y': [[df['winner'].value_counts()[team]]]}])
                for team in teams
            ],
            direction='down',
            showactive=True,
        )
    ]
)
fig5.show()

# 4. Data Storytelling

# Annotate top team in bar chart
top_team = win_counts.idxmax()
top_wins = win_counts.max()
plt.figure(figsize=(10, 6))
sns.barplot(y=win_counts.index, x=win_counts.values, palette='tab10')
plt.title('Number of Wins per Team (IPL 2023)', fontsize=16)
plt.xlabel('Wins')
plt.ylabel('Team')
plt.annotate(
    f'Top Team: {top_team} ({top_wins} wins)',
    xy=(top_wins, 0),
    xytext=(top_wins + 1, 1),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=12, color='red'
)
plt.tight_layout()
plt.show()

# Annotate close matches in scatter plot (Plotly)
fig3.add_annotation(
    x=df[df['close_match']].iloc[0]['date'],
    y=df[df['close_match']].iloc[0]['result_margin'],
    text="Example Close Match",
    showarrow=True,
    arrowhead=1
)
fig3.show()
#made by
#Vikash Kumar
#Sarvesh
#Md. Abuzer
#Abhishek Raghav