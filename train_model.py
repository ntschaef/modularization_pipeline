import run_pipeline 
from Model_Suite.model_suite import Model_Suite

args = run_pipeline.get_args()
cc,_ = run_pipeline.build_dataset(args)
ms = Model_Suite(data_source="pipeline")
ms.use_pipeline(cc)
ms.run_all_folds()
