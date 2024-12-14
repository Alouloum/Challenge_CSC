import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

scripts=['src/features_fadi.py','src/features_elie.py','src/features_combined.py','src/metamodels/meta_model1.py','src/metamodels/meta_model2.py','src/metamodels/set_difference_to_1.py']

for script in scripts:
    print("Running script: ", script)
    with open(script, 'r') as file:
        exec(file.read())
