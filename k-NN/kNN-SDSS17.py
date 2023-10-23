import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('star_classification.csv')
# Drop metadata
df = df.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID', 'MJD', 'plate'], axis = 1)
df.isnull().sum())
(df.dtypes)
# Histogram of the classes
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='class', hue='class', alpha=.7)
plt.title('Number of Instances in Each Class')
plt.show()
# Load the data again for further processing
df2 = pd.read_csv('star_classification.csv')
df2 = df2.drop(['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'fiber_ID', 'spec_obj_ID', 'MJD', 'plate'], axis = 1)
df2_no_anomaly = df2[df2['u'] > -1]
# Encode the class labels
LE = LabelEncoder()
df['class'] = LE.fit_transform(df['class'])
# Remove anomalies
df_no_anomaly = df[df['u'] > -1]
print(df_no_anomaly.head())
# Swap the last two columns
last_column = df_no_anomaly.columns[-1]
second_last_column = df_no_anomaly.columns[-2]
df_no_anomaly[last_column], df_no_anomaly[second_last_column] = df_no_anomaly[second_last_column], df_no_anomaly[last_column]
# Rename columns
df_no_anomaly.rename(columns={'redshift': 'class', 'class': 'redshift'}, inplace=True)
print(df_no_anomaly.head())
# Boxplots of stellar attributes
df2_no_anomaly.boxplot(figsize=(10, 6))
plt.title('Box Plot of Stellar Attributes')
plt.ylabel('Values')
plt.xticks(rotation=45) # Rotates x-axis labels for better visibility
plt.show()
# Remove extremely bright point and replot the box plot
df2_no_anomaly = df2[df2['u'] > -1]
df2_no_anomaly.boxplot(figsize=(10, 6))
plt.title('Box Plot of Stellar Attributes')
plt.ylabel('Values')
plt.xticks(rotation=45) # Rotates x-axis labels for better visibility
plt.show()
# Histograms of attributes
df2_no_anomaly.hist(bins = 10 , figsize= (14,14))
plt.show()
# Split data by class
galaxy = df2_no_anomaly[df2_no_anomaly['class'] == 'GALAXY']
qso = df2_no_anomaly[df2_no_anomaly['class'] == 'QSO']
star = df2_no_anomaly[df2_no_anomaly['class'] == 'STAR']
# Convert class labels to numerical values
le = LabelEncoder()
df2_no_anomaly["class"] = le.fit_transform(df2_no_anomaly["class"])
df2_no_anomaly["class"] = df2_no_anomaly["class"].astype(int)
# Prepare data for box plots
redshift = df2_no_anomaly[['redshift', 'class']]
data = [galaxy['redshift'], qso['redshift'], star['redshift']]
class_names = ['galaxy', 'qso', 'star']
colors = ['mediumslateblue', 'indianred', 'mediumaquamarine']

# Set x-tick labels using the class names
ax1.set_xticklabels(class_names)

# Fill boxes with colors
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
ax1.yaxis.grid(True)
ax1.set_ylabel('Redshift')
plt.show()

for i in range(3):
    sns.kdeplot(data=np.log(df2_no_anomaly[df2_no_anomaly["class"] == i]['redshift']), label = le.inverse_transform([i]), color=colors[i])

Classes = ['GALAXY', 'QSO', 'STAR']
plt.legend(Classes)

from matplotlib.colors import to_rgba
data = [galaxy['u'], qso['u'], star['u']]
Classes = ['GALAXY', 'QSO', 'STAR']
colors = ['mediumslateblue', 'indianred', 'mediumaquamarine']
my_colors = [to_rgba(c) for c in colors]
sns.set_palette(my_colors)

data = [galaxy['u'], qso['u'], star['u']]
Classes = ['GALAXY', 'QSO', 'STAR']
colors = ['gray', 'mediumblue', 'green'] # Updated colors
my_colors = [to_rgba(c) for c in colors]
sns.set_palette(my_colors)

def kde(feature, loc):
    for i in range(3):
        sns.kdeplot(ax=loc, data=df2_no_anomaly[df2_no_anomaly["class"] == i][feature], label=le.inverse_transform([i]), color=colors[i], linewidth=2, linestyle='--') # Changed line style
    loc.grid()
    loc.set_xlabel(feature)
    loc.set_xlim([10, 30])
    loc.set_xticks(np.arange(10, 30, 2))
    loc.legend(Classes)
    loc.spines['top'].set_visible(False) # Hide top spine
    loc.spines['right'].set_visible(False) # Hide right spine

def box(feature, loc):
    data = [galaxy[feature], qso[feature], star[feature]]
    sns.boxplot(ax=loc, data=data, palette=colors) # Use the same color palette
    loc.set_xticklabels(Classes)
    loc.set_yticks(np.arange(10, 36, 2))
    loc.set_ylabel(feature)
    loc.spines['top'].set_visible(False)
    loc.spines['right'].set_visible(False)
    loc.yaxis.grid(True)

fig, ax = plt.subplots(5, 2, figsize=(10, 18))
kde('u', ax[0, 1])
box('u', ax[0, 0])
kde('g', ax[1, 1])
box('g', ax[1, 0])
kde('r', ax[2, 1])
box('r', ax[2, 0])
kde('i', ax[3, 1])
box('i', ax[3, 0])
kde('z', ax[4, 1])
box('z', ax[4, 0])
plt.tight_layout()
plt.show()

df_no_anomaly.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit the scaler on the features and transform them
df_no_anomaly_scaled = pd.DataFrame(scaler.fit_transform(df_no_anomaly))
from sklearn.model_selection import train_test_split
# Separate features (X) and target (y)

X = df_no_anomaly_scaled.drop(columns=[7])
y = df_no_anomaly['class']
# Split the dataset into training and test sets, stratifying based on 'y'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X.head()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Create a kNN model
k = 5  # Number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = knn_model.score(X_test, y_test)
# Generate a classification report
report = classification_report(y_test, y_pred)




