import pandas as pd

df = pd.read_csv('clustered_stocks.csv')
df = df.dropna(subset=['Cluster'])

total=len(df)
cluster_sector_counts = (
    df.groupby(['Cluster', 'Sector'])
      .size()
      .reset_index(name='Count')
)

cluster_counts = (
    df['Cluster']
      .value_counts(normalize=True)
      .reset_index(name='Cluster Prior')
)
cluster_counts.columns = ['Cluster', 'Cluster Prior']
print(cluster_counts)

sector_counts = (
    df['Sector']
      .value_counts(normalize=True)
      .reset_index(name='Sector Prior')
)
sector_counts.columns = ['Sector', 'Sector_Prior']

pdf = pd.DataFrame(columns=['Sector', 'Stable Low-Risk', 'Low Volatility', 
                            'Above Average Volatility', 'Unstable High-Risk'])

cs=cluster_sector_counts
for s in cs['Sector'].unique():
    pp=[]
    for cluster in [1.0, 0.0, 2.0, 3.0]:
        Pintersection=((cs.loc[(cs['Cluster'] == cluster) & (cs['Sector'] == 
                            s)]['Count'].values[0])/total)
        Psector=sector_counts.loc[sector_counts['Sector']==s]['Sector_Prior'].values[0]
        p=Pintersection/Psector
        pp.append(p)
    pdf.loc[len(pdf)] = [s, pp[0], pp[1], pp[2], pp[3]]
all=(list(cluster_counts['Cluster Prior']))
pdf.loc[len(pdf)] = ['All Sectors', all[0], all[1], all[2], all[3]]
pdf=pdf.sort_values('Sector')
print(pdf)
pdf.to_csv('Posterior_Prob_Sector.csv', index=False)

