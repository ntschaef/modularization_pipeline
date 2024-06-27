# Parser

This will produce the raw data that other repos will pull from.

## Input
### Environmental Variable
These will override all the other variables passed.

* DATA_DB (list): Alternate list of database locations from which the database comes.
* DATA_SCHEMA (string): schema used for the databases in DATA_DB.

### Model Args
These are the variables passed through the model_args that are specific for this module.

* db_list (dict): using the date as the key, a source list of repositories from which 
                  the data will be pulled.
  * EXAMPLE - db_list: {"20220715":["KY", "LA"], "20230602": ["KY", "LA"]} 
    * EXPLINATION - this will pull data from the Kentucky and Louisiana repository 
                    from the 20220715 dataset.
* add_fields (dict): using the table as the key, this will identify additional fields that
                     can be included in the data (both for the sake of data processing and
                     future analysis.
  * EXAMPLE - add_fields: {"epath":["path_date_spec_collect_1 as pathDateSpecCollect1"],
                           "ctc":["at_risk_any_flag as atRiskAnyFlag"]}
    * EXPLINATION - this will pull pathDateSpecCollect1 and atRiskAnyFlag into the usable
                    data. Note that pathDateSpecCollect1 is already in the schema, but
                    it is captured as a "date" and will be processed as such. To get the
                    raw date at the end, the field needs to be recategorized as non-PHI.
  * NOTE: the field name are caputred in the parquet files as snake_case. Writing it as
          "field_name as fieldName" will cause the name to be captured as camelCase.
  * NOTE: A complete list of fields will be listed at the end of this README.
## Output
### Cache
These are all the files found in the generated cache

* db_schema.json (dict): the schema used to build the query. 
* deps_package (dict): the package that generates the cache
* describe.txt (str): description of what the module produces
* df_raw (pd.DataFrame[pickle in chunks]): the raw data.
* duckdb_queries.list (list): the queries used to build the dataframe using the duckdb package
* filled_files.md5 (class): the file used to test the cache for a completed build
* schema_name.txt (str): the name of the schema used.

## module calls 

* build_from_raw:   needed to run the pipeline
* get_cached_deps:  get the pregenerated arguements
* test_db:          ensure that the passed database is appropriate
* build_query:      creates the query used to generate the dataframe
* import_database:  creates the dataframe to be used in future modules
* describe:         generates a description of what was built for this module

## command line calls

using `python parser.py --<option>` you will load the database expected.

### options
This will identify the database you are choosing to capture and save at `documentation/df_raw.pkl`

  --dblist DBLIST, -db DBLIST
                        list of the (preloaded) database names being used (default: ['test'])
  --logging LOGGING, -l LOGGING
                        this is the logging level that you will see. Debug is 10 (default: 20)

## test_suite

Within the testing directory there is a test suite that will check the following:
* pipeline  call
* expected file structure
* outputs
* model arg options

This will use the files in `testing/db/` directory which includes:
* test.sqlite:          sqlite database
* test_extend.sqlite:   sqlite database

### test suite call

after navigating to the 'testing' directory
`python testing_suite.py`

## current field used
Below is the name of the fields as they appear in the parquet files. In the code they are camelCase.
### epath

#### Dates (these are converted to diffs and not a date)
* path_date_spec_collect_1

#### All fields that aren't changed
* text_path_clinical_history
* text_staging_params
* text_path_microscopic_desc
* text_path_nature_of_specimens
* text_path_supp_reports_addenda
* text_path_formal_dx
* text_path_full_text
* text_path_comments
* text_path_gross_pathology
* record_document_id
* patient_id_number
* tumor_record_number
* \_meta_filename

### ctc

#### Dates (these are converted to diffs and not a date)
* date_of_diagnosis
* rx_date_most_defin_surg
* rx_date_surgery
* date_of_birth

#### All fields that aren't changed
* \_meta_registry
* patient_id_number
* tumor_record_number
* behavior_code_icd_o_3
* grade
* histologic_type_icd_o_3
* laterality
* primary_site
* seer_summary_stage_2000
* cs_site_specific_factor_1
* cs_site_specific_factor_2
* cs_site_specific_factor_15
* cs_site_specific_factor_9
* estrogen_receptor_summary
* progesterone_recep_summary
* her_2_overall_summary
* microsatellite_instability
* kras
* cs_mets_at_dx`

### man\_coded

#### All fields that aren't changed
* \_meta_registry
* record_document_id
* reportable

## All possible fields

### epath
* registry_id
* patient_id_number
* record_document_id
* path_date_spec_collect_1
* date_epath_message
* text_path_clinical_history
* text_path_gross_pathology
* text_path_comments
* text_path_formal_dx
* text_path_nature_of_specimens
* text_path_microscopic_desc
* tumor_record_number
* text_path_supp_reports_addenda
* text_path_full_text
* text_staging_params
* \_meta_ticket
* \_meta_id
* \_meta_filename
* \_meta_src_path
* \_meta_raw_path
* \_meta_file_parsed_date
* \_meta_registry
* \_meta_collection_date
* \_meta_pipeline_git_commit
* \__null_dask_index__

### ctc
* patient_id_number
* tumor_record_number
* \_meta_registry
* race_3
* cause_of_death
* sex
* birthplace_country
* vital_status
* icd_revision_number
* place_of_death_state
* nhia_derived_hisp_origin
* birthplace_state
* date_of_birth
* place_of_death_country
* race_2
* ihs_link
* vital_status_recode
* race_5
* race_4
* date_of_last_contact
* race_1
* cs_site_specific_factor_16
* derived_ajcc_7_t
* tnm_clin_n
* laterality
* derived_ajcc_7_m_descript
* tnm_path_n
* mets_at_dx_lung
* cs_tumor_size
* rx_summ_surg_prim_site
* rx_summ_brm
* rx_summ_surg_rad_seq
* cs_tumor_size_ext_eval
* lymph_vascular_invasion
* rx_summ_reg_ln_examined
* tnm_path_stage_group
* cs_site_specific_factor_9
* rx_date_chemo
* grade_pathological
* phase_2_radiation_treatment_modality
* derived_ajcc_6_m_descript
* cs_site_specific_factor_25
* cs_site_specific_factor_24
* cs_site_specific_factor_15
* eod_mets
* state_at_dx_geocode_2000
* age_at_diagnosis
* rx_date_other
* rx_summ_systemic_sur_seq
* tnm_clin_descriptor
* derived_ajcc_6_t_descript
* recurrence_date_1st
* county_at_dx_geocode_1990
* derived_eod_2018_n
* surv_date_dx_recode
* kit_gene_immunohistochemistry
* eod_extension
* cs_lymph_nodes
* mets_at_dx_brain
* cs_lymph_nodes_eval
* surv_flag_active_followup
* behavior_code_icd_o_3
* cs_site_specific_factor_12
* surv_date_presumed_alive
* derived_eod_2018_stage_group
* tnm_clin_staged_by
* derived_ajcc_7_stage_grp
* kras
* mets_at_dx_bone
* derived_ajcc_6_stage_grp
* derived_ss_2000
* addr_at_dx_state
* tnm_clin_m
* cs_site_specific_factor_11
* rx_date_systemic
* rx_summ_chemo
* seer_cause_specific_cod
* derived_ajcc_7_m
* eod_regional_nodes
* phase_1_radiation_external_beam_tech
* cs_site_specific_factor_1
* derived_ajcc_6_n_descript
* cs_site_specific_factor_4
* derived_ajcc_7_t_descript
* derived_ajcc_6_n
* tumor_size_summary
* rx_summ_surg_oth_reg_dis
* eod_primary_tumor
* rx_date_radiation
* cs_extension
* phase_2_radiation_external_beam_tech
* cs_site_specific_factor_6
* rx_date_most_defin_surg
* census_tract_19708090
* derived_ss_2000_flag
* sentinel_lymph_nodes_examined
* rx_summ_other
* cs_site_specific_factor_13
* cs_site_specific_factor_14
* recurrence_type_1st
* phase_3_radiation_external_beam_tech
* tnm_edition_number
* cs_mets_at_dx_bone
* county_at_dx_geocode_2010
* seer_other_cod
* phase_3_radiation_treatment_modality
* rx_date_hormone
* cs_mets_at_dx_lung
* derived_summary_stage_2018
* tnm_path_t
* rx_summ_treatment_status
* mets_at_dx_other
* cs_site_specific_factor_17
* egfr_mutational_analysis
* county_at_dx
* rx_date_surgery
* derived_ss_1977_flag
* tnm_path_m
* derived_eod_2018_m
* cs_site_specific_factor_2
* type_of_reporting_source
* record_number_recode
* eod_lymph_node_involv
* surv_date_active_followup
* grade_clinical
* alk_rearrangement
* grade
* tumor_size_pathologic
* primary_site
* rx_summ_surg_oth_9802
* mets_at_dx_liver
* date_initial_rx_seer
* derived_ajcc_7_n
* histologic_type_icd_o_3
* county_at_dx_geocode_2000
* cs_site_specific_factor_8
* cs_site_specific_factor_21
* rx_summ_hormone
* eod_tumor_size
* cs_site_specific_factor_5
* surv_flag_presumed_alive
* derived_ss_1977
* cs_site_specific_factor_18
* surv_mos_presumed_alive
* cs_site_specific_factor_23
* cs_site_specific_factor_3
* tnm_path_descriptor
* microsatellite_instability
* progesterone_recep_summary
* sequence_number_central
* cs_mets_at_dx_liver
* census_tract_2010
* state_at_dx_geocode_19708090
* tnm_clin_stage_group
* cs_mets_eval
* regional_nodes_positive
* mets_at_dx_distant_ln
* derived_ajcc_flag
* census_tr_certainty_2000
* primary_payer_at_dx
* rx_summ_scope_reg_9802
* cs_mets_at_dx_brain
* derived_eod_2018_t
* seer_summary_stage_2000
* derived_ajcc_7_n_descript
* phase_1_radiation_treatment_modality
* tumor_size_clinical
* cs_site_specific_factor_22
* rx_summ_radiation
* date_of_diagnosis
* eod_extension_prost_path
* tnm_path_staged_by
* census_at_dx_geocode_2010
* state_at_dx_geocode_2010
* derived_ajcc_6_m
* census_tr_cert_19708090
* her2_overall_summary
* icd_o_3_conversion_flag
* surv_mos_active_followup
* cs_site_specific_factor_20
* sentinel_lymph_nodes_positive
* cs_site_specific_factor_19
* cs_site_specific_factor_7
* census_tract_2000
* census_cod_sys_19708090
* rx_summ_scope_reg_ln_sur
* regional_nodes_examined
* cs_site_specific_factor_10
* reason_for_no_surgery
* rx_summ_surgical_margins
* estrogen_receptor_summary
* derived_ajcc_6_t
* rx_summ_transplnt_endocr
* summary_stage_2018
* reason_for_no_radiation
* rx_date_brm
* cs_mets_at_dx
* tnm_clin_t
* county_at_dx_analysis
* recurrence_date_1st_flag
* date_initial_rx_seer_flag
* rx_date_radiation_flag
* rx_date_chemo_flag
* rx_date_hormone_flag
* rx_date_brm_flag
* rx_date_surgery_flag
* rx_date_most_defin_surg_flag
* rx_date_systemic_flag
* rx_date_other_flag
* \_meta_ticket
* \_meta_id
* \_meta_filename
* \_meta_src_path
* \_meta_raw_path
* \_meta_file_parsed_date
* \_meta_collection_date
* \_meta_pipeline_git_commit

### man_coded
* record_document_id
* reportable
* \_meta_ticket
* \_meta_id
* \_meta_registry
* \_meta_collection_date
* \_meta_filename
* \_meta_src_path
* \_meta_file_parsed_date
* \_meta_raw_path
* \_meta_pipeline_git_commit
* \__null_dask_index__
