from wrangler import Wrangler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
        
    print("loading data")
    pdf = Wrangler.load_pickle("data/pdf.pkl")
    nndf = Wrangler.load_pickle("data/nndf.pkl")

    print("splitting data")
    #splitting data from the pdf dataframe
    pdf_train,pdf_test = train_test_split(pdf, test_size=0.2, random_state=1)
    
    #splitting data from the nndf dataframe
    train,test = train_test_split(np.unique(nndf["index"]), test_size=0.2, random_state=1)
    mask = np.isin(nndf["index"],train)
    nndf_train = nndf.loc[mask]
    nndf_test = nndf.loc[~mask]

    print("Split done. Dumping new files")

    #dumping pdf files
    Wrangler.dump_pickle(pdf_train,"data/pdf_train.pkl")
    Wrangler.dump_pickle(pdf_test,"data/pdf_test.pkl")

    #dumping nndf files
    Wrangler.dump_pickle(nndf_train,"data/nndf_train.pkl")
    Wrangler.dump_pickle(nndf_test,"data/nndf_test.pkl")