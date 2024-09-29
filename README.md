# **SWARM Inference Module**

[![Upload Python Package](https://github.com/saandeepa93/SWARM_lib/actions/workflows/python-publish.yml/badge.svg)](https://github.com/saandeepa93/SWARM_lib/actions/workflows/python-publish.yml)

## **Setup**

```
conda create -n <env-name> --file requirements.txt
```

  + Ensure to fill in your \<env-name>

## **Inference**
```
python inference.py --dir <dir_path>
```

+ `<dir_path>` should be the path to set of csv files for which you'd like to compute the predictions

+ The final csv file with predictions will be saved under `./data/<dir_name>.csv`

