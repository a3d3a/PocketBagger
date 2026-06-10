#import packages
import datetime
rundate=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="PocketBagger script")
parser.add_argument("--development", type=lambda x: x.lower() == "true", default=True, help="Controls development mode")
parser.add_argument("--experiment", type=str, default='shorttest', help="Set experiment name")
parser.add_argument("--isotrees", type=int, default=500, help="Set number of isolation forest trees")
parser.add_argument("--bootstrap", type=int, default=10000, help="Set number of bootstap samples")
parser.add_argument("--writemodels", type=lambda x: x.lower() == "true", default=False, help="Write out models")
parser.add_argument("--writepredictions", type=lambda x: x.lower() == "true", default=False, help="Write out models")
parser.add_argument("--runholdouttests", type=lambda x: x.lower() == "true", default=False, help="Write out models")

args = parser.parse_args()

model_version = '3.0.0' #model_type.dataset_change.minor_tweak 
development = args.development
experiment=args.experiment
iso_trees = args.isotrees
n_boot = args.bootstrap
foldername = f'{experiment}_{rundate}'

os.makedirs(foldername, exist_ok=True)

try:
    # Copy file, preserving metadata
    shutil.copy2(sys.argv[0], foldername)
    print(f"File '{sys.argv[0]}' copied (with metadata) to '{foldername}'.")
except Exception as e:
    print(f"An error occurred: {e}")


print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
ACOCSTRING=f'#A3D3A_DCOC {rundate}: '+" ".join(sys.argv)+'\n'
print(ACOCSTRING)
print(args)

#machine learning libraries
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, recall_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit, PredefinedSplit, cross_val_predict, StratifiedKFold, cross_val_score, StratifiedGroupKFold
from skopt.space import Integer, Categorical, Real
from skopt import BayesSearchCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from scipy.stats import spearmanr

#standard utilities
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
import multiprocessing as mp
import pickle
import gzip
import copy
import warnings

from wordcloud import WordCloud

#filter some warnings
warnings.filterwarnings("ignore",message="The objective has been evaluated at*",category=UserWarning)
warnings.filterwarnings("ignore", message="X has feature names, but DecisionTreeRegressor was fitted without feature names")
warnings.filterwarnings("ignore", message="The groups parameter is ignored*")
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores*")

# define an early  stopper callback
class NoImprovementStopper:
    def __init__(self, patience=5, skip_initial=None, verbose = False):
        """
        patience: how many *counted* iterations in a row with no improvement before stopping
        skip_initial: how many *first* iterations to ignore entirely (if None, defaults to 10)
        """
        self.patience     = patience
        self.skip_initial = skip_initial or 10
        self.iteration    = 0
        self.best_score   = -np.inf
        self.bad_iters    = 0
        self.verbose = verbose
    
    def __call__(self, result):
        # 1) count total callbacks
        self.iteration += 1
        current_score = -result.func_vals[-1]
        
        # 3) skip the warm-up, but collect the best score in the warmup
        if self.iteration <= self.skip_initial:
            if current_score > self.best_score:
                self.best_score = current_score
            return False
        
        # 4) check for improvement
        if current_score > self.best_score:
            self.best_score = current_score
            #self.patience = max(1,self.patience - 1)
            self.bad_iters  = 0
        else:
            self.bad_iters += 1
            if self.verbose:
                print(f"No improvement for {self.bad_iters}/{self.patience} iterations.")
        
        # 5) stop if we’ve gone 'patience' rounds without bettering our best
        if self.bad_iters >= self.patience:
            if self.verbose:
                print("Stopping early due to no improvement.")
            return True
        
        return False

def area_between_curves_df(test, num_thresholds=500):
    """
    Compute ∫_0^1 [R(t) - U(t)] dt using dataframe columns:
      - test['label'] : 1 for positive, 0 for unlabeled
      - test['prob']  : predicted probabilities
    """

    thresholds = np.linspace(0, 1, num_thresholds)

    pos_scores = test.loc[test['label'] == 1, 'prob'].values
    unl_scores = test.loc[test['label'] == 0, 'prob'].values

    R = np.zeros_like(thresholds)  # recall among known positives
    U = np.zeros_like(thresholds)  # fraction of unlabeled predicted positive

    for i, t in enumerate(thresholds):
        R[i] = np.mean(pos_scores >= t)
        U[i] = np.mean(unl_scores >= t)

    # Numeric integration (trapezoidal rule)
    area = np.trapezoid(R - U, thresholds)

    return area, thresholds, R, U

def rho_scorer(estimator, X, y):
    #custom callable scoring function to facilitate feature importance for isolation forests
    new_scores = estimator.score_samples(X)
    rho, _ = spearmanr(new_scores, y)
    return rho

def plot_pu_curves(thresholds, R, U):
    #plt.figure(figsize=(7, 5))
    plt.plot(thresholds, R, label="Recall among positives R(t)")
    plt.plot(thresholds, U, label="Fraction unlabeled predicted positive U(t)")

    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("PU-Learning Separation Curves")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

#drug_like ligands
drug_like_file = 'reasonable_ligands_20260218.tsv'
print(f'File containing reasonable ligand HET codes used for labeling positives: {drug_like_file}')
drug_like_hets=pd.read_csv(drug_like_file,sep='\t')

# Convert drug-like HETs to a set for fast lookup
drug_like_hets = set(drug_like_hets["HET"].tolist() + ['LIG']) #we add LIG as that is what is used in XCHEM and doesn't appear in the PDB

# Function to check if any HET in a row is drug-like
def contains_drug_like_het(het_string):
    het_list = het_string.split(',')  # Split into individual HET codes
    return int(any(het in drug_like_hets for het in het_list))  # Label 1 if any match, else 0

#import clusters from mmseqs of the entire pdbseqres
def cap_pdbid_chain(pdbid_chain):
    pdbid=pdbid_chain.split('_')[0].upper()
    chain=pdbid_chain.split('_')[1]
    return pdbid+'_'+chain

clusters=pd.read_csv('pdb_clusterRes_20260220_cluster.tsv',names=['cluster','pdbid_chain'],sep='\t')
for column in clusters.columns.tolist():
    clusters[column]=clusters[column].apply(lambda x: cap_pdbid_chain(x))

#new pdbids for grasp test
new_pdbs = pd.read_csv('pdbid_chains_after_2023026.tsv',sep='\t')
new_pdbs = new_pdbs.merge(clusters,on='pdbid_chain', how='left').dropna().reset_index(drop=True)

#read in lists of key drug target families to drive targeted holdout tests later
PKs=pd.read_csv('PK_list.txt',names=['uniprot_id']).uniprot_id.tolist()
NHRs=pd.read_csv('nhr_list.txt',names=['uniprot_id']).uniprot_id.tolist()
GPCRs=pd.read_csv('gpcr_list.txt',names=['uniprot_id']).uniprot_id.tolist()

#load in the grasp test set and assign clusters
grasp_test = pd.read_csv('grasp_set_predictions_20260302_123228.tsv',sep='\t')#.drop(columns=['cluster'])
#grasp_test = grasp_test.merge(clusters, on='pdbid_chain', how='left').dropna()
#grasp_test = grasp_test[grasp_test.het.isin(drug_like_hets)].reset_index(drop=True)
#grasp_test = grasp_test.rename(columns={'het':'lig_codes'})

#read in xchem data, and label the pockets
xchem = pd.read_csv('xchem_features.csv')
xchem['occupied']=1
xchem.loc[xchem.lig_codes.isna(),'occupied']=0
xchem['label']=0
xchem.loc[xchem.lig_codes.isna() == False, "label"] = xchem[xchem.lig_codes.isna() == False]["lig_codes"].apply(contains_drug_like_het)

#import the dataset
input_file='20260217_surfnet_pockets.tsv.gz' # '20250311_surfnet_pockets.tsv.gz' # 
print(f'Input pocket data file: {input_file}')
df=pd.read_csv(input_file,sep='\t',low_memory=False).sample(frac=1,random_state=0)

feature_cols_orig=['acc_bur_vert_ratio',
              'acc_ratio',
              'acc_vertices',
              'andrews_energy',
              'beta_sheet',
              'bur_vertices',
              'cons_rating',
              'hp_ratio',
              'hb_acceptor',
              'hb_both',
              'hb_donor',
              'helix',
              'hot_fraction',
              'long_axis',
              'loop',
              'max_depth',
              'mean_axis',
              'nrm_hyd_ratio',
              'nrm_polar_ratio',
              'pca_x',
              'pca_y',
              'pca_z',
              'pocket_size',
              'turn',
              'vol_ratio']

df=df.rename(columns={'feature_labels':'label','db_id':'uniprot_id','new_column':'lig_codes','druggability':'druggability_old'})

# label the pockets according to the presence of a drug-like ligand
df['label']=0
df.loc[df.lig_codes.isna() == False, "label"] = df[df.lig_codes.isna() == False]["lig_codes"].apply(contains_drug_like_het)

#label pockets as occupied or not
df['occupied']=1
df.loc[df.lig_codes.isna(),'occupied']=0

#get pdbid_chain cluster assignments
df = df.merge(clusters, on='pdbid_chain', how='left') #merge in pdb cluster info

#remove NaNs
df=df.dropna(subset=feature_cols_orig+['chain_id','pdbid_chain','uniprot_id','cluster','label']).reset_index(drop=True)

#setup grasp hold out splits
#grasp_test=grasp_test.merge(df,how='left',on=['pdbid_chain','lig_codes']).drop_duplicates(subset=['pdbid_chain','lig_codes'], keep = False).dropna().reset_index(drop=True)
df['GrASP']=-1
df.loc[df.pdbid_chain.isin(grasp_test.pdbid_chain),'GrASP']=0

#setup family holdout splits
df['PK']=-1
df.loc[df.uniprot_id.isin(PKs),'PK']=0
df['NHR']=-1
df.loc[df.uniprot_id.isin(NHRs),'NHR']=0
df['GPCR']=-1
df.loc[df.uniprot_id.isin(GPCRs),'GPCR']=0

#for development, downsample the unlabeled data simply for speed.
if development:
    df = df.groupby('label',as_index = False).head(df.label.value_counts().min()).reset_index(drop=True)

print(f'Total positively labeled pockets: {df.label.sum()}')
print(f"{df['pdbid_chain'].nunique()} unique pdbid_chains")
print(f"{df['cluster'].nunique()} unique sequence clusters")

#print('Running feature reduction')
feature_cols=feature_cols_orig.copy()
print('Features:',feature_cols)

nhr_ps=PredefinedSplit(df['NHR'].values)
pks_ps=PredefinedSplit(df['PK'].values)
gpcrs_ps=PredefinedSplit(df['GPCR'].values)
grasp_ps=PredefinedSplit(df['GrASP'].values)

print(f'\nExperiment: {experiment}')

#setup the classifier
decisiontree = DecisionTreeClassifier(criterion='log_loss', random_state=0)
extratree = ExtraTreeClassifier(criterion='gini', random_state=0)
clfmodel = BalancedBaggingClassifier(estimator= decisiontree,
                                     n_estimators= 100 if development else 200,
                                     replacement=False,
                                     bootstrap=False,
                                     n_jobs=-1,
                                     random_state=0)

#overfit a model to set a ceiling on the max depth we'll explore
print('Fitting a model to overfit')
clf=clone(clfmodel)
clf.estimator.max_features = 'sqrt'
clf.fit(df[feature_cols],df['label'])
depths = [tree.named_steps['classifier'].get_depth() for tree in clf.estimators_] #BalancedBaggingClassifier with DecisionTree or ExtraTree Classifier
#importances = [tree.named_steps['classifier'].feature_importances_ for tree in clf.estimators_] #BalancedBaggingClassifier with DecisionTree or ExtraTree Classifier
#importances = np.vstack(importances).mean(axis=0)
#importances = pd.DataFrame(importances, columns=['importance'])
#importances['feature'] = feature_cols
#print(importances.sort_values(by='importance',ascending = False))
#feature_cols = importances.sort_values(by='importance',ascending = False)['feature'].tolist()
upper_depth_limit = max(depths)
del clf
del depths

#set the search space for depth
print(f'Setting the maximum tree depth to {upper_depth_limit}')
depth_space = Integer(1, upper_depth_limit, prior="log-uniform")  

#setup cv splitting options
cvs = 5 # of folds for data splitters; drive the % holdout with this
test_cvs = (1 if development else 5) #drive the number of times to loop through the test folds
val_cvs = (1 if development else 5) #drive the number of times to loop through the val folds
sgkf = StratifiedGroupKFold(n_splits=cvs,shuffle=True,random_state=0)
skf = StratifiedKFold(n_splits=cvs,shuffle=True,random_state=0)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

clf_list = [extratree] if development else [extratree, decisiontree]

bayes = BayesSearchCV(
    estimator = clfmodel,
    search_spaces={#'estimator': Categorical(clf_list),
                   'estimator__max_depth': depth_space,
                   'estimator__max_features': Integer(1,len(feature_cols),prior='log-uniform'),
                   #'estimator__random_state': [0],
                   #'estimator__criterion': Categorical(['log_loss','gini']),
                   },
    scoring='average_precision', #this is a ranking metric, so given the PU nature of the problem, we'll tune against this
    cv=sss,
    n_iter=10 if development else 1000, 
    random_state=42,
    n_jobs=1, 
    return_train_score = False,
    refit=True,
    optimizer_kwargs = {'n_initial_points': 5 if development else 10,'initial_point_generator':'lhs'},
)

print(f'Hyperparameter tuning score: {bayes.scoring}')

#recalls
recall_scores_overall=[]
recall_scores_in=[]
recall_scores_out=[]

#areas between pos and unlabeled prediction curves
pusas = []
pusas_overall = []
pusas_in = []
pusas_out =[]

#recall@ki - for structure i with k true positives, how many do we get in our top k predictions per structure?
recall_kis_overall = []
recall_kis_out = []

overall_count = []
inlier_count =[]
outlier_count=[]

split_type = []

split_dict = {'Random': skf,
              'Chain-Aware':sgkf,
              'Protein-Aware':sgkf,
              'Cluster-Aware':sgkf,
              'Permutation Test':sgkf,
              'Protein Kinase':pks_ps,
              'NHR':nhr_ps,
              'GPCR':gpcrs_ps,
              'GrASP':grasp_ps,
}

grasp_recall_df = pd.DataFrame(columns=['Recall','model','Tier'])
for splitter_name in split_dict.keys():
    if args.runholdouttests == False:
        break
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Running {splitter_name} hold out sets...')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    splitter = split_dict[splitter_name]
    if splitter_name == 'Cluster-Aware':
        group_label='cluster'
    elif splitter_name == 'Protein-Aware':
        group_label='uniprot_id'
    else:
        group_label='pdbid_chain'
    
    view_IDH1 = True
    for fold, (train_index, test_index) in tqdm(enumerate(splitter.split(df[feature_cols], df['label'], groups=df[group_label].values)), total=test_cvs):
        if fold == test_cvs:
            break
        split_type.append(splitter_name)
        print(f"Fold {fold}:")
        test = df.iloc[test_index].copy()
        train = df.iloc[train_index].copy()
        
        #drop training records in clusters occupied by targeted holdouts in certain test sets
        if splitter_name in ['NHR','Protein Kinase','GPCR','GrASP']:
            print(f'Dropping training records in clusters occupied by {splitter_name} structures')
            train = train[train.cluster.isin(test.cluster) == False].copy()
            if splitter_name == 'GrASP':
                train = train[(train.pdbid_chain.isin(new_pdbs.pdbid_chain) == False)].copy()
        
        print(f"  Train: {len(train)} total records | {len(train[train.label == 1])} total positives | {len(train.drop_duplicates('pdbid_chain'))} total chains | {len(train.drop_duplicates('cluster'))} total clusters")
        print(f"  Test:  {len(test)} total records | {len(test[test.label == 1])} total positives | {len(test.drop_duplicates('pdbid_chain'))} total chains | {len(test.drop_duplicates('cluster'))} total clusters")
        
        #confirm no identical records are in the train and test set
        print(len(train[train.index.isin(test.index)]), "test set records in the training set")
        
        if len(train[train.cluster.isin(test.cluster)]) > 0:
            print('LEAKAGE WARNGING: Records belonging to the same sequence clusters span train and test sets.')
        else:
            print('No records belonging to the same sequence clusters span train and test sets.')
        
        #setup the validation folds for hyperparameter tuning
        train['val_split']=-1
        for val_fold, (_, val_index) in enumerate(sgkf.split(train[feature_cols], train['label'], groups=train.cluster.values)):
            if val_fold == val_cvs:
                break
            train.loc[train.index[val_index],'val_split']=val_fold
        
        ps=PredefinedSplit(train.val_split.values)
        
        #setup permutation test
        if splitter_name == 'Permutation Test':
            print('Performing permutation test.')
            train['label']=train.sample(frac=1,random_state=0)['label'].values
        
        #select the features to be used
        selected_feats = feature_cols.copy()
        
        print(f'Performing hyperparameter tuning for fold {fold}')
        hp_search = clone(bayes)
        hp_search.cv=ps
        stopper = NoImprovementStopper(patience=5, skip_initial=hp_search.optimizer_kwargs.get('n_initial_points', 5))
        hp_search.fit(train[selected_feats],train['label'],groups=train['cluster'].values,callback=stopper)
        best_score = hp_search.best_score_ #.cv_results_['mean_test_score'][hp_search.best_index_]
        best_params = hp_search.best_params_
        clf=hp_search.best_estimator_
                
        #predict on the test set
        test['prob']=clf.predict_proba(test[selected_feats])[:,1]
        test['pred']=np.where(test.prob.values >= 0.5, 1, 0)
        
        #train isolation forest, weighting samples by cluster weights
        print('Training Isolation Forest')
        cluster_sizes = train[train.label == 1]["cluster"].value_counts()
        sample_weights = train[train.label == 1]["cluster"].map(lambda c: 1 / cluster_sizes[c])
        outclf = IsolationForest(n_estimators=iso_trees,n_jobs=-1,contamination='auto',random_state=0,bootstrap=True)
        rho = 0.0
        max_samples=0
        #increase the number of max samples until rho stabilizes over 0.99
        while rho < 0.99 and max_samples < len(train[train.label == 1]):
            #max_samples = max_samples + 256
            max_samples = min(max_samples + 256, len(train[train.label == 1]))
            outclf.max_samples = max_samples
            outclf.random_state = 10
            outclf.fit(train[train['label'] == 1][selected_feats], sample_weight = sample_weights)
            scores1=outclf.score_samples(train[train['label'] == 1][selected_feats])
            outclf.random_state = 0
            outclf.fit(train[train['label'] == 1][selected_feats], sample_weight = sample_weights)
            scores=outclf.score_samples(train[train['label'] == 1][selected_feats])
            rho, _ = spearmanr(scores1, scores)
        
        print(f'Setting max samples for IsolationForest to {max_samples}')
        rng = np.random.default_rng(0)  # choose any fixed integer seed
        boot_scores = rng.choice(scores, size=(n_boot, len(scores)), replace=True)
        boot_offsets = np.percentile(boot_scores,25, axis=1) - 1.5*(np.percentile(boot_scores,75, axis=1) - np.percentile(boot_scores,25, axis=1))
        orig_contam = len(scores[scores < -0.5])/len(scores)
        lower_bound = np.median(boot_offsets)
        tukey_contam = (boot_scores < boot_offsets.reshape(-1,1)).mean(axis=1).mean()
        #tukey_contam = np.round(len(scores[scores < lower_bound])/len(scores),3)
        contam = min(0.01, orig_contam) if tukey_contam <= 0 else tukey_contam #max(0.01, tukey_contam)#, orig_contam)
        print(f'Original Contamination: {orig_contam:.3f}. Tukey-based Contamination: {tukey_contam:.3f} with a lower bound of {lower_bound:.3f}. Adjusting Isolation Forest contamination to {contam:.3f}.')
        outclf.contamination = contam
        outclf.fit(train[train['label'] == 1][selected_feats], sample_weight = sample_weights)
        train['IF_SCORE'] = outclf.decision_function(train[selected_feats])
        train['in_out'] = np.where(train.IF_SCORE < 0, -1,1)
        
        #apply the isolation forest
        test['IF_SCORE']=outclf.decision_function(test[selected_feats])
        test['in_out']=np.where(test.IF_SCORE < 0, -1,1)
        
        #spit out a few results to punch into pymol for visualization
        if len(test[(test.uniprot_id == 'O75874')]) > 0 and view_IDH1 == True:
            print(test[(test.uniprot_id == 'O75874') & (test.label == 0) & (test.pred == 1) & (test.lig_codes.str.contains('-') == False) & (test.num_ligands == 1)].sample(frac=1).head(20).sort_values(by='prob')[['site_num','pdbid_chain','prob','pred','IF_SCORE','label','lig_codes']])
            view_IDH1 = False
        else:
            print('No IDH1 structures with druggable residues in this test fold')
        
        print(test[(test.label == 0) & (test.pred == 1) & (test.lig_codes.str.contains('-') == False) & (test.num_ligands == 1)].sample(frac=1).head(20).sort_values(by='prob')[['site_num','pdbid_chain','prob','pred','IF_SCORE','label','lig_codes']])
        
        print('Best Score:', np.round(best_score,5), '| Best Params:', best_params)
        if best_params['estimator__max_depth'] == upper_depth_limit and splitter_name != 'Permutation Test':
            print('WARNGING: Best model depth = upper depth limit.  Greater tree depths may be needed.')
        
        print(classification_report(test.label.values,test.pred.values))
        
        #compute recall@Ki per structure
        def compute_recall_ki_for_struc(struc):
            # access global df "test"
            tempdf = test[test.pdbid_chain == struc]

            k = (tempdf.label == 1).sum()
            if k == 0:
                return None   # will filter out later

            recalled = (
                tempdf.sort_values(by='prob', ascending=False)
                  .head(k)['label']
                  .sum()
            )
            return recalled / k
        
        # ----- parallel section -----
        strucs = test.pdbid_chain.unique().tolist()

        with mp.Pool() as pool:
            recall_kis = list(tqdm(pool.imap(compute_recall_ki_for_struc, strucs,chunksize = min(10,len(strucs))),total=len(strucs), miniters=len(strucs) // 10, mininterval=0, maxinterval=float('inf')))

        # remove None values
        recall_kis = [r for r in recall_kis if r is not None]
        recall_kis_overall.append(np.mean(recall_kis))
        
        #get the PU Separability Area
        #area, thresholds, R, U = area_between_curves_df(test)
        #pusas.append(area)
        #plot_pu_curves(thresholds, R, U)
        
        #generate inlier and outlier set
        inliers=test[test.in_out == 1]
        outliers=test[test.in_out == -1]
    
        overall_count.append(len(test[test['label'] == 1]))
        inlier_count.append(len(inliers[inliers['label'] == 1]))
        outlier_count.append(len(outliers[outliers['label'] == 1]))
        
        #generate grasp data - this needs to be done here to get the inlier/outlier labels
        if splitter_name == 'GrASP':
            grasp_test=grasp_test.merge(test[test.label == 1][selected_feats+['pdbid_chain','site_num','prob','pred','in_out','IF_SCORE']],how='left',on=['pdbid_chain','site_num'])
            #sns.histplot(grasp_test,x='prob',hue=grasp_test.in_out.astype(str),binrange=(0,1),bins=20)
            #plt.tight_layout()
            #plt.savefig(f'{foldername}/{experiment}_grasp_probs.png',bbox_inches='tight')
            #plt.show()
            #sns.scatterplot(grasp_test,x='IF_SCORE',y='prob',hue=grasp_test.in_out.astype(str))
            #plt.tight_layout()
            #plt.savefig(f'{foldername}/{experiment}_grasp_probs.png',bbox_inches='tight')
            #plt.show()
            grasp_test.to_csv(f'{foldername}/{experiment}_grasp_predictions.tsv',sep='\t',index=None)
            #print(grasp_test[['label','pred','prob','in_out']].describe())
            grasp_mean_recall={'Recall':(grasp_test.adj_prob_mean >= 0.3).mean(),'model':'GrASP (Site Mean)','Tier':'Overall'}
            grasp_mean_inlier_recall={'Recall':(grasp_test[grasp_test.in_out == 1].adj_prob_mean >= 0.3).mean(),'model':'GrASP (Site Mean)','Tier':'Inliers'}
            grasp_mean_outlier_recall={'Recall':(grasp_test[grasp_test.in_out == -1].adj_prob_mean >= 0.3).mean(),'model':'GrASP (Site Mean)','Tier':'Outliers'}
            grasp_max_recall={'Recall':(grasp_test.adj_prob_max >= 0.3).mean(),'model':'GrASP (Site Max)','Tier':'Overall'}
            grasp_max_inlier_recall={'Recall':(grasp_test[grasp_test.in_out == 1].adj_prob_max >= 0.3).mean(),'model':'GrASP (Site Max)','Tier':'Inliers'}
            grasp_max_outlier_recall={'Recall':(grasp_test[grasp_test.in_out == -1].adj_prob_max >= 0.3).mean(),'model':'GrASP (Site Max)','Tier':'Outliers'}
            grasp_PocketBagger_recall={'Recall':recall_score(grasp_test.label.values,grasp_test.pred.values),'model':'PocketBagger','Tier':'Overall'}
            grasp_PocketBagger_inlier_recall={'Recall': recall_score(grasp_test[grasp_test.in_out == 1].label.values,grasp_test[grasp_test.in_out == 1].pred.values) if (grasp_test.in_out == 1).sum() > 0 else np.nan,'model':'PocketBagger','Tier':'Inliers'}
            grasp_PocketBagger_outlier_recall={'Recall': recall_score(grasp_test[grasp_test.in_out == -1].label.values,grasp_test[grasp_test.in_out == -1].pred.values) if (grasp_test.in_out == -1).sum() > 0 else np.nan,'model':'PocketBagger','Tier':'Outliers'}
            
            #write out the recall of the GrASP sites
            #grasp_recall_df = pd.DataFrame(columns=['Recall','model','Tier'])
            for result_dict in [grasp_PocketBagger_recall, grasp_PocketBagger_inlier_recall, grasp_PocketBagger_outlier_recall, grasp_mean_recall, grasp_mean_inlier_recall, grasp_mean_outlier_recall, grasp_max_recall, grasp_max_inlier_recall, grasp_max_outlier_recall]:
                grasp_recall_df.loc[len(grasp_recall_df)] = result_dict

            grasp_recall_df.to_csv(f'{foldername}/{experiment}_grasp_comparison.tsv',sep='\t',index=None)
            
            train['Set']='Training Inliers'
            train.loc[train.in_out == -1, 'Set']='Training Outliers'
            scaler = MinMaxScaler() #StandardScaler()
            scaler.fit(train[train.label == 1][selected_feats])
            grasp_test['Set']='GrASP Test Holdout Inliers'
            grasp_test.loc[grasp_test.in_out == -1, 'Set']='GrASP Test Holdout Outliers'
            temp_df = pd.concat([train[train.label == 1][selected_feats +['Set']], grasp_test[selected_feats + ['Set']]])
            temp_df[selected_feats] = scaler.transform(temp_df[selected_feats])
            for feat in selected_feats:
                sns.histplot(temp_df,x=feat,hue='Set',stat='percent',common_norm=False)
                plt.tight_layout()
                plt.savefig(f'{foldername}/{experiment}_{feat}_grasp.png',bbox_inches='tight')
                plt.cla()
                plt.clf()
            
            del temp_df
        
        #append recall by tier
        for score_list, results in [(recall_scores_overall,test),(recall_scores_in,inliers),(recall_scores_out,outliers)]:
            try:
                score_list.append(recall_score(results.label.values,results.pred.values))
            except:
                score_list.append(np.nan)
        
        #append pusas by tier
        for score_list, results in [(pusas_overall,test),(pusas_in,inliers),(pusas_out,outliers)]:
            try:
                area, _, _, _ = area_between_curves_df(results)
                score_list.append(area)
            except:
                score_list.append(np.nan)
    
    #break
                
#create a bar plot of the segmented results
overall=pd.DataFrame({'Recall':recall_scores_overall,'Recall@Ki':recall_kis_overall,'Positive-Unlabeled Separation Area':pusas_overall,'Split Type':split_type,'Sample Count':overall_count,'Tier':'Overall'})
tier_in=pd.DataFrame({'Recall':recall_scores_in,'Positive-Unlabeled Separation Area':pusas_in,'Split Type':split_type,'Sample Count':inlier_count,'Tier':f'Inliers'})
tier_out=pd.DataFrame({'Recall':recall_scores_out,'Positive-Unlabeled Separation Area':pusas_out,'Split Type':split_type,'Sample Count':outlier_count,'Tier':f'Outliers'})

#tiers=pd.concat([oob_overall, oob_tier_in, oob_tier_out,overall,tier_in,tier_out]).reset_index(drop=True).round(3)
tiers=pd.concat([overall,tier_in,tier_out]).reset_index(drop=True).round(3)
tiers['model']='PU Bagger'
tiers.to_csv(f'{foldername}/{experiment}_performance.tsv',sep='\t',index=None)

#print a table of the means
for splitter_name in split_dict.keys():
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Overall {splitter_name} Split Performance')
    print(tiers[tiers['Split Type'] == splitter_name].groupby('Tier').mean(numeric_only=True).round(3))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(f'GrASP Comparison Results')
print(grasp_recall_df[['Recall','Tier','model']])
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
sns.barplot(grasp_recall_df,x='Tier',y='Recall',hue='model',capsize=0.05,err_kws={'linewidth': 2}, errorbar = 'sd')
plt.xlabel('Segment')
plt.title(f'PocketBagger vs. GrASP True Positive Recall (n = {len(grasp_test)} pockets)')
plt.ylim(0,1.05)
plt.legend(title='Model',loc='lower left')#, fontsize=8)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_grasp_comparison.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[tiers['Sample Count'].notna()],x='Tier',y='Sample Count',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Sample Counts (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.legend(title='Experiment', loc='upper right')
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_sample_count_by_tier.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[tiers['Recall'].notna()],x='Tier',y='Recall',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Nested CV True Positive Recall (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.legend(title='Experiment', loc='lower left', ncols=3)#, fontsize=10)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_recall_by_tier.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[(tiers['Recall'].notna()) & (tiers['Split Type'].isin(['Random','Chain-Aware','Protein-Aware','Cluster-Aware','Permutation Test']))],x='Tier',y='Recall',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Nested CV True Positive Recall (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.legend(title='Experiment', loc='lower left', ncols=2)#, fontsize=8)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_nestedcv_recall_by_tier.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[(tiers['Recall'].notna()) & (tiers['Split Type'].isin(['NHR','Protein Kinase','GPCR']))],x='Tier',y='Recall',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Protein Family Holdout True Positive Recall')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.legend(title='Protein Family', loc='lower left')#, fontsize=8)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_familyholdouts_recall_by_tier.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[tiers['Recall@Ki'].notna()],x='Split Type',y='Recall@Ki',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Nested CV Recall@Ki (n = {test_cvs} repeats)')
plt.xlabel('Experiment')
plt.ylim(0,1.05)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_recall_ki.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[(tiers['Recall@Ki'].notna()) & (tiers['Split Type'].isin(['Random','Chain-Aware','Protein-Aware','Cluster-Aware','Permutation Test']))],x='Split Type',y='Recall@Ki',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Nested CV Recall@Ki (n = {test_cvs} repeats)')
plt.xlabel('Experiment')
plt.ylim(0,1.05)
#plt.xticks(labels=['Random','Chain-Aware','Protein-Aware','Cluster-Aware','Permutation\nTest'])#rotation=90)
labels = [t.get_text() for t in ax.get_xticklabels()]
labels = [l.replace('Permutation Test', 'Permutation\nTest') for l in labels]
ax.set_xticklabels(labels)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_nestedcv_recall_ki.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[(tiers['Recall@Ki'].notna()) & (tiers['Split Type'].isin(['Protein Kinase','NHR','GPCR']))],x='Split Type',y='Recall@Ki',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Protein Family Holdout Recall@Ki')
plt.xlabel('Protein Family')
plt.ylim(0,1.05)
#plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_familyholdouts_recall_ki.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[tiers['Positive-Unlabeled Separation Area'].notna()],x='Tier',y='Positive-Unlabeled Separation Area',capsize=0.05,err_kws={'linewidth': 2},hue='Split Type', errorbar = 'sd')
plt.title(f'Nested CV PUSA (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.legend(title = 'Experiment', loc='upper left', ncols=3) #fontsize=8)
#plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_pusas.png', bbox_inches="tight")
plt.cla()
plt.clf()

#train production model
train = df.copy().reset_index(drop=True)
train['Set']='Training Unlabeled'
train.loc[train.label == 1, 'Set']='Training Positives'

#setup the validation splits
print(f'Performing hyperparameter tuning for the final model')
train['val_split']=-1
for val_fold, (_, val_index) in enumerate(sgkf.split(train[feature_cols], train['label'], groups=train.cluster.values)):
    if val_fold == val_cvs:
        break
    train.loc[train.index[val_index], 'val_split'] = val_fold

ps=PredefinedSplit(train.val_split.values)

#select the features to be used
selected_feats = feature_cols.copy()

#train isolation forest, weighting samples by cluster weights
print('Training Isolation Forest')
cluster_sizes = train[train.label == 1]["cluster"].value_counts()
sample_weights = train[train.label == 1]["cluster"].map(lambda c: 1 / cluster_sizes[c])
outclf = IsolationForest(n_estimators=iso_trees,n_jobs=-1,contamination='auto',random_state=0,bootstrap=True)
rho = 0.0
max_samples=0
#increase the number of max samples until rho stabilizes over 0.99
while rho < 0.99 and max_samples < len(train[train.label == 1]):
    max_samples = max_samples + 256
    outclf.max_samples = max_samples
    outclf.random_state = 10
    outclf.fit(train[train['label'] == 1][selected_feats], sample_weight = sample_weights)
    scores1=outclf.score_samples(train[train['label'] == 1][selected_feats])
    outclf.random_state = 0
    outclf.fit(train[train['label'] == 1][selected_feats], sample_weight = sample_weights)
    scores=outclf.score_samples(train[train['label'] == 1][selected_feats])
    rho, _ = spearmanr(scores1, scores)

print(f'Setting max samples for IsolationForest to {max_samples}')
rng = np.random.default_rng(0)  # choose any fixed integer seed
boot_scores = rng.choice(scores, size=(n_boot, len(scores)), replace=True)
boot_offsets = np.percentile(boot_scores,25, axis=1) - 1.5*(np.percentile(boot_scores,75, axis=1) - np.percentile(boot_scores,25, axis=1))
orig_contam = len(scores[scores < -0.5])/len(scores)
lower_bound = np.median(boot_offsets)
tukey_contam = (boot_scores < boot_offsets.reshape(-1,1)).mean(axis=1).mean()
#tukey_contam = np.round(len(scores[scores < lower_bound])/len(scores),3)
contam = min(0.01, orig_contam) if tukey_contam <= 0 else tukey_contam #max(0.01, tukey_contam)#, orig_contam)
print(f'Original Contamination: {orig_contam:.3f}. Tukey-based Contamination: {tukey_contam:.3f} with a lower bound of {lower_bound:.3f}. Adjusting Isolation Forest contamination to {contam:.3f}.')
outclf.contamination = contam
outclf.fit(train[train['label'] == 1][selected_feats], sample_weight = sample_weights)

#apply the isolation forest
train['IF_SCORE']=outclf.decision_function(train[selected_feats])
train['in_out']=np.where(train.IF_SCORE < 0, -1,1)

#compute the permutation importance of features for the isolation forest by measuring the change in rho between initial scores and scores after feature permutation
outclf_sensitivity = pd.DataFrame(selected_feats,columns=['feature'])
outclf_sensitivity['outclf_sensitivity']=permutation_importance(outclf, train[train.label ==1][selected_feats], train[train.label ==1]['IF_SCORE'], scoring=rho_scorer,n_repeats= 1 if development else 5, random_state=0,n_jobs=1).importances_mean

hp_search = clone(bayes)
hp_search.cv=ps
stopper = NoImprovementStopper(patience=5, skip_initial=hp_search.optimizer_kwargs.get('n_initial_points', 5))
hp_search.fit(train[selected_feats],train['label'],groups=train['cluster'].values,callback=stopper)
best_score = hp_search.best_score_ #.cv_results_['mean_test_score'][hp_search.best_index_]
best_params = hp_search.best_params_
clf=hp_search.best_estimator_
print('Best Score:', np.round(best_score,3), '| Best Params:', best_params)

#let's predict on the xchem dataset
xchem['Set']='XChem Unlabeled'
xchem.loc[xchem.label == 1, 'Set']='XChem Positives'
xchem['IF_SCORE'] = outclf.decision_function(xchem[selected_feats])
xchem['in_out'] = np.where(xchem.IF_SCORE < 0, -1,1)
xchem['prob'] = clf.predict_proba(xchem[selected_feats])[:,-1]
xchem['pred'] = np.where(xchem['prob'] < 0.5, 0, 1)
xchem['IF_SCORE'] = outclf.decision_function(xchem[selected_feats])
xchem['in_out'] = np.where(xchem.IF_SCORE < 0, -1,1)
xchem_in = xchem[xchem.in_out == 1]
xchem_out = xchem[xchem.in_out == -1]

xchem_recall_overall = []
xchem_recall_in = []
xchem_recall_out = []
xchem_recall_kis =[]

#compute recall@Ki per structure
def compute_recall_ki_for_struc(struc):
    # access global df "xchem"
    tempdf = xchem[xchem.pdbid_chain == struc]

    k = (tempdf.label == 1).sum()
    if k == 0:
        return None   # will filter out later

    recalled = (
        tempdf.sort_values(by='prob', ascending=False)
          .head(k)['label']
          .sum()
    )
    return recalled / k

# ----- parallel section -----
strucs = xchem.pdbid_chain.unique().tolist()

with mp.Pool() as pool:
    recall_kis = list(tqdm(pool.imap(compute_recall_ki_for_struc, strucs,chunksize = min(10,len(strucs))),total=len(strucs), miniters=len(strucs) // 10, mininterval=0, maxinterval=float('inf')))

# remove None values
recall_kis = [r for r in recall_kis if r is not None]
xchem_recall_kis.append(np.mean(recall_kis))

#append recall by tier
for score_list, results in [(xchem_recall_overall,xchem),(xchem_recall_in,xchem_in),(xchem_recall_out,xchem_out)]:
    try:
        score_list.append(recall_score(results.label.values,results.pred.values))
    except:
        score_list.append(np.nan)

#create a bar plot of the segmented results
xchem_overall=pd.DataFrame({'Recall':xchem_recall_overall,'Recall@Ki':xchem_recall_kis,'Sample Count':len(xchem),'Tier':'Overall'})
xchem_in=pd.DataFrame({'Recall':xchem_recall_in,'Sample Count':len(xchem_in),'Tier':f'Inliers'})
xchem_out=pd.DataFrame({'Recall':xchem_recall_out,'Sample Count':len(xchem_out),'Tier':f'Outliers'})

#tiers=pd.concat([oob_overall, oob_tier_in, oob_tier_out,overall,tier_in,tier_out]).reset_index(drop=True).round(3)
xchem_results=pd.concat([xchem_overall,xchem_in,xchem_out]).reset_index(drop=True).round(3)
xchem_results['model']='PU Bagger'
xchem_results.to_csv(f'{foldername}/{experiment}_xchem_performance.tsv',sep='\t',index=None)
xchem.to_csv(f'{foldername}/{experiment}_xchem_features_predictions.tsv',sep='\t',index=None)

print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(f'XChem Prediction Results')
print(xchem_results)

#plot chem results
sns.barplot(xchem_results,x='Tier',y='Recall',capsize=0.05,err_kws={'linewidth': 2}, errorbar = 'sd')
plt.xlabel('Segment')
plt.title(f'True Positive Recall for XChem Structures (n = {len(xchem[xchem.label == 1])} pockets)')
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_xchem_recall.png', bbox_inches="tight")
plt.cla()
plt.clf()

sns.histplot(pd.concat([train[train.label == 1],xchem[xchem.label == 1]],axis=0),x='IF_SCORE',hue='Set',bins=20,stat='percent',common_norm=False)
plt.xlabel('Isolation Score Relative to Known Positives\n(Lower = More Anomalous)')
plt.title('Anomaly Score Comparison')
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_xchem_anomaly_comparison.png', bbox_inches="tight")
plt.cla()
plt.clf()

#perform a bootstrap test of the xchem positve IF scores vs the training dataset to see if the means are different
train_pos = train.loc[train.label == 1, "IF_SCORE"].dropna().to_numpy()
xchem_pos = xchem.loc[xchem.label == 1, "IF_SCORE"].dropna().to_numpy()

obs = train_pos.mean() - xchem_pos.mean()
print(f'Observed Isolation Forest score effect size between Training Positives and XChem Positives: {obs:.4f}')

# Generate bootstrap resamples
print('Perform bootstrap comparison of the means of the training and XChem sets according to isolation forest scores')
rng = np.random.default_rng(0)  # choose any fixed integer seed
train_samples = rng.choice(train_pos, size=(n_boot, len(train_pos)), replace=True)
xchem_samples = rng.choice(xchem_pos, size=(n_boot, len(xchem_pos)), replace=True)

# Compute bootstrap means
train_means = train_samples.mean(axis=1)
xchem_means = xchem_samples.mean(axis=1)

# Difference distribution
diffs = train_means - xchem_means
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
#p_value = 2 * min(
#    (diffs <= 0).mean(),
#    (diffs >= 0).mean()
#)
p_value = (diffs <= 0).mean()
print(f'95% Confidence Interval: {ci_low:.4f}, {ci_high:.4f}')
print("One-sided Bootstrap Mean Comparison p-value:", p_value)

#plot the distribution
sns.histplot(diffs, bins=20)
plt.xlabel('Mean Difference of Anomaly Score Distributions')
plt.title(f'Bootstrap Mean Comparison of Positive Pocket Anomaly Score Distributions\nbetween Training and XChem Datasets (n = {n_boot} samples)')
plt.axvline(x=ci_low, color='gray', linestyle='dashed', linewidth=2)
plt.axvline(x=ci_high, color='gray', linestyle='dashed', linewidth=2)
# Add legend entry for the vertical line
handles = [plt.Line2D([0], [0], color='gray', linestyle='dashed', lw=2)]
labels = ["95% CI"]

plt.legend(handles=handles, labels=labels, loc='upper left')#, fontsize=8)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_xchem_anomaly_bootstrap_comparison.png', bbox_inches="tight")
plt.cla()
plt.clf()

print('\nCreating out of fold predictions, bootstrapped model uncertainty estimatates, and feature importance')
sgkf.random_state = 100 #change the random state to just mix things up a bit
for val_fold, (train_index, val_index) in enumerate(tqdm(sgkf.split(train[selected_feats], train['label'], groups=train.cluster.values), total=sgkf.get_n_splits())):
    #generate temp datasets
    temp_train = train.iloc[train_index].copy()
    temp_val = train.iloc[val_index].copy()
    
    #train the classifier for this fold
    clf_fold = clone(clf)
    clf_fold.fit(temp_train[selected_feats],temp_train['label'])
    
    #make predictions on temp_val
    train.loc[train.index[val_index],'prob']=clf_fold.predict_proba(temp_val[selected_feats])[:,1]
        
    #run permutation importance on the val fold
    temp_importance = pd.DataFrame(selected_feats,columns=['feature'])
    
    imp = permutation_importance(clf_fold, temp_val[selected_feats], temp_val['label'], scoring='average_precision',n_repeats= 1 if development else 5, random_state=0, n_jobs=1).importances_mean
    
    temp_importance['importance'] = imp
    
    if val_fold == 0:
        importance_PU = temp_importance.copy()
    else:
        importance_PU = pd.concat([importance_PU,temp_importance]).reset_index(drop=True)
    
    #make bootstrap predictions for the training data to have them for later data mining:
    probs = []
    for n in range(10 if development else 100):
        sub_train = temp_train.groupby('label', as_index=False).sample(frac=1,replace=True,random_state=n)
        sub_clf = clone(clf)
        sub_clf.set_params(estimator__random_state=n)
        sub_clf.fit(sub_train[selected_feats],sub_train['label'])
        probs.append(sub_clf.predict_proba(temp_val[selected_feats])[:, -1])
        del sub_train
        del sub_clf

    probs = np.vstack(probs)
    mean_probs = probs.mean(axis=0)
    median_probs = np.median(probs, axis=0)
    ci_lower = np.percentile(probs, 2.5,axis=0)
    ci_upper = np.percentile(probs, 97.5,axis=0)
    
    train.loc[train.index[val_index],'prob_avg'] = mean_probs
    train.loc[train.index[val_index],'prob_median'] = median_probs
    train.loc[train.index[val_index],'prob_CI_lower'] = ci_lower
    train.loc[train.index[val_index],'prob_CI_upper'] = ci_upper
    train.loc[train.index[val_index],'prob_CI_width'] = ci_upper - ci_lower
    
    del temp_train
    del temp_val
    del temp_importance
    del probs
    del mean_probs
    del median_probs
    del ci_lower
    del ci_upper

train['pred']=np.where(train.prob >= 0.5, 1, 0)

#write out predictions
if args.writepredictions == True:
    output_file = input_file.split('.')[0]+f'_PocketBaggerPredictions_{rundate}.tsv.gz'
    train.to_csv(f'{foldername}/' + output_file,sep='\t',index=None)

print('Sample OOF predictions:')
print(train.groupby('label',as_index = False).sample(20).sort_values(by=['label','prob'],ascending = [False,False])[['pdbid_chain','site_num','label','prob','prob_avg','prob_median','prob_CI_lower','IF_SCORE']].round(3))

#let's cluster the features based on pairwise linear correlations
corr = 1 - train[selected_feats].corr().pow(2)

def sil_score(clusterer, X, y=None):
    metric = clusterer.metric
    if 1 < len(np.unique(clusterer.labels_)) < len(X):
        score = silhouette_score(X, clusterer.labels_, metric=metric)
    else:
        score = 0.0
    return score

clusterer = AgglomerativeClustering(distance_threshold=0.1,n_clusters=None, metric="precomputed", linkage='average')
clustering = BayesSearchCV(estimator = clusterer, 
                           search_spaces= {'distance_threshold':Real(0.0,1.0)}, 
                           cv=[(np.array(range(len(corr))),np.array(range(len(corr))))],
                           n_iter=10 if development else 1000,
                           n_jobs=1,
                           refit=True,
                           scoring=sil_score,
                           verbose=False,
                           optimizer_kwargs = {'n_initial_points': 5 if development else 10,'initial_point_generator':'lhs'},
                          )

stopper = NoImprovementStopper(patience=10, skip_initial=clustering.optimizer_kwargs.get('n_initial_points', 5), verbose=True)
clustering.fit(corr,callback=stopper)
print(f'Best feature clustering threshold: {clustering.best_params_['distance_threshold']} | Best feature clustering silhouette score: {clustering.best_score_}')

for n, feat in enumerate(selected_feats):
    importance_PU.loc[importance_PU.feature == feat, 'cluster'] = clustering.best_estimator_.labels_[n]

#save out feature importance info and plot it
importance_PU = importance_PU.sort_values(by=['cluster','importance','feature'],ascending = [True,False,False]).reset_index(drop=True)
importance_PU = importance_PU.merge(outclf_sensitivity, on='feature', how='left')

"""
print(importance_PU)
importance_PU.to_csv(f'{foldername}/{experiment}_PocketBagger_feature_importance.tsv',sep='\t',index=None)
#print(importance_PU.groupby('feature').mean().sort_values('cluster',ascending=True))

importance_PU['cluster'] = importance_PU['cluster'].astype(int)
importance_PU['feature_cluster'] = importance_PU['feature'] + ' (' + importance_PU['cluster'].astype(str) + ')'

order = (
    importance_PU.sort_values(by=['cluster','importance'], ascending=[True, False])
    .drop_duplicates(subset=['feature_cluster'])
    ['feature_cluster']
    .tolist()
)

importance_PU['cluster'] = importance_PU['cluster'].astype(str)

sns.barplot(data=importance_PU, x='feature_cluster',y='importance',capsize=0.05,err_kws={'linewidth': 2}, order=order, hue='cluster', errorbar = 'sd')
plt.xlabel('Feature (Cluster)')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.title('PocketBagger Feature Importance')
plt.ylim(0,)
plt.legend(title='Cluster', fontsize=8, ncol=4)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_PocketBagger_feature_importance.png', bbox_inches="tight")
plt.cla()
plt.clf()
"""

# --- reorder clusters by the most important feature in each cluster ---

# use mean importance per feature across folds/repeats if importance_PU has repeated rows
feature_summary = (
    importance_PU.groupby(['feature', 'cluster'], as_index=False)
    .agg(importance=('importance', 'mean'))
)

# score each cluster by its top feature importance
cluster_rank_df = (
    feature_summary.groupby('cluster', as_index=False)
    .agg(cluster_top_importance=('importance', 'max'))
    .sort_values(by='cluster_top_importance', ascending=False)
    .reset_index(drop=True)
)

# map old cluster labels -> new ordered cluster labels: 0,1,2,...
cluster_map = {old_cluster: new_cluster
               for new_cluster, old_cluster in enumerate(cluster_rank_df['cluster'])}

importance_PU['cluster_ordered'] = importance_PU['cluster'].map(cluster_map)

# if you want the saved/displayed cluster IDs to reflect this new ordering:
importance_PU['cluster'] = importance_PU['cluster_ordered'].astype(int)
importance_PU.drop(columns=['cluster_ordered'],inplace=True)

importance_PU = importance_PU.sort_values(by=['cluster', 'importance'], ascending=[True, False]).reset_index(drop=True)
print(importance_PU)

importance_PU.to_csv(f'{foldername}/{experiment}_PocketBagger_feature_importance.tsv',
                     sep='\t', index=None)

importance_PU['feature_cluster'] = (
    importance_PU['feature'] + ' (' + importance_PU['cluster'].astype(str) + ')'
)

# order clusters by ordered cluster id, then members by descending importance
order = (
    importance_PU.groupby(['feature_cluster', 'cluster'], as_index=False)
    .agg(importance=('importance', 'mean'))
    .sort_values(by=['cluster', 'importance'], ascending=[True, False])
    ['feature_cluster']
    .tolist()
)

importance_PU['cluster'] = importance_PU['cluster'].astype(str)

sns.barplot(
    data=importance_PU,
    x='feature_cluster',
    y='importance',
    capsize=0.05,
    err_kws={'linewidth': 2},
    order=order,
    hue='cluster',
    errorbar='sd'
)
plt.xlabel('Feature (Cluster)')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.title('PocketBagger Feature Importance')
plt.ylim(0,)
plt.legend(title='Cluster', fontsize=8, ncol=4)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_PocketBagger_feature_importance.png', bbox_inches="tight")
plt.cla()
plt.clf()

#generate a word cloud based on feature importance
importance_PU_mean = importance_PU.groupby('feature',as_index=False)[['importance','outclf_sensitivity']].mean().sort_values('importance',ascending=False)
print(importance_PU_mean[['feature','importance','outclf_sensitivity']])
importance_PU_mean['frequency'] = importance_PU_mean['importance'] / importance_PU_mean['importance'].max()
freqs = dict(zip(importance_PU_mean["feature"], importance_PU_mean["frequency"]))
wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate_from_frequencies(freqs)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_wordcloug.png', bbox_inches="tight")
plt.cla()
plt.clf()

sns.barplot(data=importance_PU_mean.sort_values(by=['outclf_sensitivity'],ascending=[False]), x='feature',y='outclf_sensitivity')
plt.xlabel('Feature')
plt.ylabel('Feature Sensitivity')
plt.xticks(rotation=90)
plt.title('Isolation Forest Feature Permutation Sensitivity')
plt.ylim(0,)
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_PocketBagger_outclf_feature_sensitivity.png', bbox_inches="tight")
plt.cla()
plt.clf()

#train a classifier to a get the isolation forest score that best separates positives and unlabeled
# Create the 2D histogram plot of IF scores and OOB probabilities to visualize cutoffs that may be useful in production
#fit a classifier on the in_out_scores and druggability labels to visualize a boundary in the 2D histplot
clf_in_out=BalancedRandomForestClassifier(max_depth=1,class_weight=None,random_state=0,n_estimators=100,max_features=None,replacement=False,bootstrap=False,n_jobs=-1)
clf_in_out.fit(train[['IF_SCORE']],train['label'])
train['pred_from_in_out_score']=clf_in_out.predict(train[['IF_SCORE']])
in_out_boundary = np.mean([train[train.pred_from_in_out_score == 0].IF_SCORE.max(),train[train.pred_from_in_out_score == 1].IF_SCORE.min()])
print(f'Strong positives should have an isolation forest anomaly score of >= {in_out_boundary}')

#estimate the classification threshold
#k = (train[train.label == 1].prob.quantile(0.25) - train[train.label == 0].prob.quantile(0.75))/((train[train.label == 0].prob.quantile(0.75) - train[train.label == 0].prob.quantile(0.25)) + (train[train.label == 1].prob.quantile(0.75) - train[train.label == 1].prob.quantile(0.25)))
#pos_tukey = train[train.label == 1].prob.quantile(0.25) - k*(train[train.label == 1].prob.quantile(0.75) - train[train.label == 1].prob.quantile(0.25))
#unl_tukey = train[train.label == 0].prob.quantile(0.75) + k*(train[train.label == 0].prob.quantile(0.75) - train[train.label == 0].prob.quantile(0.25))
#class_threshold = min(0.5,np.mean([pos_tukey,unl_tukey]))
#print('Class threshold:',class_threshold, 'Positive tukey:', pos_tukey, 'Unlabeled Tukey:', unl_tukey, 'Tukey Average:',np.mean([pos_tukey,unl_tukey]), 'Tukey K:', k)

#for k in [3.0,2.2,1.5]:
#    pos_tukey = train[train.label == 1].prob.quantile(0.25) - k*(train[train.label == 1].prob.quantile(0.75) - train[train.label == 1].prob.quantile(0.25))
#    unl_tukey = train[train.label == 0].prob.quantile(0.75) + k*(train[train.label == 0].prob.quantile(0.75) - train[train.label == 0].prob.quantile(0.25))
#    print(pos_tukey, unl_tukey)
    
unl_tukey15 = train[train.label == 0].prob.quantile(0.75) + 1.5*(train[train.label == 0].prob.quantile(0.75) - train[train.label == 0].prob.quantile(0.25))
unl_tukey22 = train[train.label == 0].prob.quantile(0.75) + 2.2*(train[train.label == 0].prob.quantile(0.75) - train[train.label == 0].prob.quantile(0.25))
unl_tukey30 = train[train.label == 0].prob.quantile(0.75) + 3.0*(train[train.label == 0].prob.quantile(0.75) - train[train.label == 0].prob.quantile(0.25))

#train an Elkan and Noto classifier to estimate the class prior
print('Fitting a EN-classifier model to overfit')
en_clf = RandomForestClassifier(n_estimators = 100 if development else 250, class_weight=None, random_state=42, criterion = 'gini' if development else 'log_loss', n_jobs =-1)

cluster_sizes = train["cluster"].value_counts()
train["cluster_weight"] = train["cluster"].map(lambda c: 1 / cluster_sizes[c])

temp_clf = clone(en_clf)
temp_clf.fit(train[selected_feats],train['label'],sample_weight=train.cluster_weight.values)
depths = [tree.get_depth() for tree in temp_clf.estimators_]
upper_depth_limit = max(depths)
del temp_clf
del depths

#set the search space for depth
print("Max depth for EN Classifier:", upper_depth_limit)
depth_space = Integer(1, upper_depth_limit, prior="log-uniform")  

calibrated_clf = CalibratedClassifierCV(
    en_clf,
    method="isotonic",
    cv= sss,
    ensemble='auto'
)

en_bayes = BayesSearchCV(
    estimator = calibrated_clf,
    search_spaces={'estimator__max_depth': depth_space},
    scoring='neg_log_loss',
    cv = sgkf,
    n_iter=100, 
    random_state=42,
    n_jobs=1, 
    return_train_score = False,
    refit=True,
    optimizer_kwargs = {'n_initial_points':5,'initial_point_generator':'lhs'},
)

print(f'Performing hyperparameter tuning for the EN model')
hp_search = clone(en_bayes)
hp_search.cv = ps
stopper = NoImprovementStopper(patience=5, skip_initial=hp_search.optimizer_kwargs.get('n_initial_points', 5),verbose=False)
hp_search.fit(train[selected_feats],train['label'], groups=train.cluster.values,sample_weight=train.cluster_weight.values,callback=stopper)
best_score = hp_search.best_score_ #.cv_results_['mean_test_score'][hp_search.best_index_]
print('Best Score:', np.round(best_score,3), '| Best Params:', hp_search.best_params_)
en_clf=hp_search.best_estimator_

#generate out of fold probabilities, where folds are group aware
#cluster weights are used to try to reduce how badly we're breaking the SCAR assumption
train['EN_prob'] = cross_val_predict(
    en_clf,
    train[selected_feats],
    train['label'],
    method='predict_proba',
    cv=sgkf,
    groups=train.cluster.values,
    params={'sample_weight':train.cluster_weight.values},
    n_jobs=1
)[:, 1]

#estimate the class prior
c_hat = train[train['label'] == 1]['EN_prob'].mean()
estimated_class_prior = train['EN_prob'].mean() / c_hat

print("Estimated c:", c_hat)
print("EN-estimated class prior:", estimated_class_prior)

EN_thresh = train['prob'].quantile(1-estimated_class_prior)
print(f'Elkan and Noto derived positive classification threshold: {EN_thresh}')

# use a tukey fence to limit how much of the outliers is shown - some of them have quite a low score make the plot less meaningful
# Create the 2D histogram plot
lower_limit=train.IF_SCORE.quantile(0.25) - 6*(train.IF_SCORE.quantile(0.75)-train.IF_SCORE.quantile(0.25))
lower_limit=max(train.IF_SCORE.min(),lower_limit)

hue_categories = np.where(train.sort_values(by='prob',ascending=True).pred.unique() == 1, 'Positive','Negative').tolist()

palette = sns.color_palette()[:len(hue_categories)]
palette.reverse()
plot = sns.histplot(train[train.IF_SCORE >= lower_limit].sort_values(by='prob',ascending=True), x='IF_SCORE', y='prob', hue=train[train.IF_SCORE >= lower_limit].sort_values(by='prob',ascending=True).pred.astype(str), binwidth=(0.001,0.05),palette=palette,common_norm=False,cbar=True,cbar_kws={"label": "Bin Count"})

# Customize the labels
plt.xlabel('Isolation Score Relative to Known Positives\n(Lower = More Anomalous)')
plt.ylabel('OOF Prediction Score')

# Add a vertical dashed line at x = 0.0 to indicate the outlier threshold
plt.axvline(x=0.0, color='black', linestyle='dashed', linewidth=2)
plt.axvline(x=in_out_boundary, color='gray', linestyle='dashed', linewidth=2)
plt.axhline(y=EN_thresh, color='blue', linestyle='dashed', linewidth=2)
plt.axhline(y=unl_tukey15, color='red', linestyle='dashed', linewidth=2)
plt.axhline(y=unl_tukey30, color='green', linestyle='dashed', linewidth=2)

# Manually construct the legend
handles = [plt.Line2D([0], [0], color=palette[i], lw=5) for i in range(len(hue_categories))]
labels = hue_categories

# Add legend entry for the vertical line
handles.append(plt.Line2D([0], [0], color='black', linestyle='dashed', lw=2))
labels.append("Positive Outlier\nThreshold")

# Add legend entry for the vertical line
handles.append(plt.Line2D([0], [0], color='gray', linestyle='dashed', lw=2))
labels.append("Positive-Negative IF\nThreshold")

# Add legend entry for the vertical line
handles.append(plt.Line2D([0], [0], color='blue', linestyle='dashed', lw=2))
labels.append("Positive-Negative EN\nThreshold")

# Add legend entry for the vertical line
handles.append(plt.Line2D([0], [0], color='red', linestyle='dashed', lw=2))
labels.append("Unlabeled Tukey\nThreshold (k=1.5)")

# Add legend entry for the vertical line
handles.append(plt.Line2D([0], [0], color='green', linestyle='dashed', lw=2))
labels.append("Unlabeled Tukey\nThreshold (k=3.0)")

plt.legend(handles=handles, labels=labels, loc='upper left')#, fontsize=8)
plt.title(f'OOF Scores vs Anomaly Scores')
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_IFscores_OOFscores.png',bbox_inches='tight')
plt.cla()
plt.clf()

#generate a plot of probabilities
train['Class']=np.where(train.label == 1, 'Positive', 'Unlabeled')
sns.histplot(train,x='prob',hue='Class', binrange=(0,1), bins=20, stat='percent', common_norm = False)
plt.title('OOF Prediction Scores')
plt.xlabel('Score')
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_scores.png', bbox_inches="tight")
plt.cla()
plt.clf()

#save out the final models
if args.writemodels == True:  

    with gzip.open(f'{foldername}/PocketBagger_singlemodels_{experiment}_{rundate}.pkl.gz','wb') as f:
        pickle.dump({"druggability_model":clf,
                     "isolation_forest":outclf,
                     "features":selected_feats,
                     'low_threshold':unl_tukey15,
                     'mid_threshold':unl_tukey22,
                     'high_threshold':unl_tukey30,
                     'EN_threshold':EN_thresh,
                     'isoforest_threshold':in_out_boundary,
                     'date':rundate,
                     'model_version':model_version},f)
        
    #train multiple models built on bootstrap samples of the training data and save each
    print('Training multiple models for uncertainty in production')
    clfs=[]
    for n in tqdm(range(10 if development else 100)):
        sub_train = train.groupby('label', as_index=False).sample(frac=1,replace=True,random_state=n)
        sub_clf = clone(clf)
        sub_clf.set_params(estimator__random_state=n)
        sub_clf.fit(sub_train[selected_feats],sub_train['label'].values)
        clfs.append(sub_clf)

    with gzip.open(f'{foldername}/PocketBagger_bootstrapmodels_{experiment}_{rundate}.pkl.gz','wb') as f:
        pickle.dump({'bootstrap_clfs':clfs,
                     'features':selected_feats,
                     'date':rundate,
                     'model_version':model_version},f)