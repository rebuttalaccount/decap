## Dataset Preparation

Make sure your dataset is organized in the following structure:

```
/datasetname/
└── val/
    └── classname/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Each subdirectory under `val/` should be named after the class it represents and contain the corresponding images.

## How to Run

1. Prepare the dataset by running the following script:

```bash
bash prepare.sh
```

2. Then, start the main program:

```
python main.py
```
