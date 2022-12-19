# otto

````
├── LICENSE
├── Pipfile           
├── README.md          
├── data
│   ├── raw             <- Raw data in various formats
│   ├── candidates      <- Ready to load candidates in `parquet` format
│   └── features        <- Ready to use features for the model
└── otto 
    ├── cross_val.py    <- Main code
    ├── candidates.py   <- Loads candidates files, if needed generates candidates
    ├── features.py     <- Loads features files, if needed generates features
    ├── rerank_model.py <- Runs rerank model and scores candidates
    └── evaluate.py     <- Evaluates solution
 ````
