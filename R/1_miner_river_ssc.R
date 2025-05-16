#### i. LIBRARY IMPORTS ####
## Tables
library(data.table)
library(readxl)
library(rgdal)
library(lubridate)
library(tidyr)
library(broom)

## Plots
library(ggplot2)
library(maps)
library(scales)
library(ggthemes)
library(ggpubr)
library(gstat)
library(markdown)
library(ggtext)
library(patchwork)
library(egg)
library(zoo)

## Data download
library(dataRetrieval)
library(tidyhydat)

## Analysis
library(glmnet)
library(Hmisc)

#### ii. THEMES ####
theme_evan <- theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(linetype = 'dashed',color = 'grey70'),
    panel.grid.major.x = element_blank(),
    # panel.grid = element_blank(),
    legend.position = 'none',
    panel.border = element_rect(size = 0.5),
    text = element_text(size=8),
    axis.text = element_text(size = 8), 
    plot.title = element_text(size = 9)
  )

theme_evan_facet <- theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_blank(),
    # panel.grid = element_blank(),
    # legend.position = 'none',
    panel.border = element_rect(size = 0.5),
    strip.background = element_rect(fill = 'white'),
    text = element_text(size=10),
    axis.text = element_text(size = 10), 
    plot.title = element_text(size = 13)
  )
season_facet <- theme_evan_facet + theme(
  legend.position = 'none', 
  strip.background = element_blank(),
  strip.text = element_text(hjust = 0, margin = margin(0,0,0,0, unit = 'pt'))
)

theme_dark_mode <- season_facet +
  theme(
    axis.title.x = element_markdown(color = 'white'),
    axis.title.y = element_markdown(color = 'white'),
    strip.text.x.top = element_markdown(color = 'white'),
    panel.grid.major.x = element_line(color = 'grey90', linewidth = 0.25),
    panel.grid.major.y = element_line(color = 'grey90', linewidth = 0.25),
    panel.grid.minor.x = element_line(color = 'grey90', linewidth = 0.1),
    panel.grid.minor.y = element_line(color = 'grey90', linewidth = 0.1),
    text = element_text(color = 'white'),
    panel.background = element_rect(fill = 'black'),
    plot.background = element_rect(fill = 'black'),
    axis.text = element_markdown(color = 'white'),
    panel.border = element_rect(color = 'white'),
    legend.background = element_rect(color = 'white', fill = 'black'),
    legend.key = element_rect(fill = 'black')
  ) 

fancy_scientific_modified <- function(l) { 
  # turn in to character string in scientific notation 
  if(abs(max(log10(l), na.rm = T) - min(log10(l), na.rm = T)) > 2 | 
     # min(l, na.rm = T) < 0.01 | 
     max(l, na.rm = T) > 1e5){ 
    l <- log10(l)
    label <- parse(text = paste("10^",as.character(l),sep = ""))
  }else{
    label <- parse(text = paste(as.character(l), sep = ""))
  }
  # print(label)
  # return(parse(text=paste("'Discharge [m'", "^3* s", "^-1 ", "*']'", sep="")))
  return(label)
}

lat_dd_lab <- function(l){
  label <- c()
  for(i in 1:length(l)){
    label_sel <- ifelse(l[i] < 0, paste0(abs(l[i]), '°S'), 
                        paste0(abs(l[i]), '°N'))
    label <- c(label, label_sel)
  }
  return(label)}

long_dd_lab <- function(l){
  label <- c()
  for(i in 1:length(l)){
    label_sel <- ifelse(l[i] < 0, paste0(abs(l[i]), '°W'), 
                        paste0(abs(l[i]), '°E'))
    label <- c(label, label_sel)
  }
  return(label)}

abbrev_year <- function(l){
  label <- c() 
  for(i in 1:length(l)){
    label_sel <- paste0("'",substr(as.character(l[i]),3,4))
    label <- c(label, label_sel)  
  }
  return(label)}

# Custom labeller function
positive_latitude_labeller <- function(x) {
  return(as.character(abs(as.numeric(x))))
}


#### iii. SET DIRECTORIES ####
# Set root directory
wd_root <- getwd()

# Imports folder (store all import files here)
wd_R <- paste0(wd_root,"/R/")
wd_imports <- paste0(wd_root,"/imports/")

wd_figures <- paste0(wd_root, "/figures/")
wd_dod_data <- paste0(wd_imports,'slump_arctic_dem_dods/')

# Create folders within root directory to organize outputs if those folders do not exist
export_folder_paths <- c(wd_R, wd_imports, wd_figures,wd_dod_data)
for(i in 1:length(export_folder_paths)){
  path_sel <- export_folder_paths[i]
  if(!dir.exists(path_sel)){
    dir.create(path_sel)}
}


#### 1A. IMPORT DATA AND DEFINE COLUMNS ####

# Set region name for exports
region_name <- 'arctic_thaw_slump'

# Import Landsat river profile data for each batch of thaw slump sites
# Combine Landsat sample data into one data.table
river_import <- rbindlist(
  lapply(paste0(wd_imports, 
                c('arctic_thaw_slump_wRef_training_ls5789_rawBands_b7lt500.csv',
                  'arctic_thaw_slump_wRef_training_ls5789_rawBands_b7lt500_2.csv')
  ),
  fread
  ), fill = T, use.names = T)[
    ,':='(.geo = NULL)]

# Remove slumps s005 and s006
river_import <- river_import[!grepl('005|006', site_no)]

river_import <- rbind(river_import, 
                      fread(paste0(wd_imports, 'arctic_thaw_slump_wRef_training_ls5789_rawBands_b7lt500_3.csv'))[,':='(.geo = NULL)],
                      fill = T, use.names = T)

cloud_cover <- river_import[,.(site_no = paste0('st_', site_no), sample_dt = date, cloud_cover, cloud_qa_count, snow_ice_qa_count)]
#### 1B. PREPARE DATA COLUMNS, MAKE LANDSAT MATCHUP, CALCULATE SSC ESTIMATE ####
# Import Landsat data
# **Takes about three minutes**

# Landsat data do have station information
# They also have latitude and longitude
river_import <- na.omit(river_import[,
                                     ':='(
                                       site_no = paste0('st_', site_no),
                                       # site_no = name,
                                       station_nm = paste0('st_', site_no),
                                       # Rename columns for simplicity
                                       B1 = B1_median,
                                       B2 = B2_median,
                                       B3 = B3_median,
                                       B4 = B4_median,
                                       B5 = B5_median,
                                       B6 = B6_median,
                                       B7 = B7_median,
                                       num_pix = B2_count,
                                       sample_dt = ymd(date),
                                       landsat_dt = ymd(date)
                                     )]
                        , cols = c('B1','B2','B3','B4','B5','B7'))[
                          B1 > 0 & B2 > 0 & B3 > 0 & B4 > 0 & B5 > 0 & B7 > 0 &
                            B1 < 5000 & B2 < 5000 & B3 < 5000 & B4 < 5000 & B6 < 4000][
                              ,':='( 
                                year = year(sample_dt),
                                # add new columns with band ratios
                                B1.2 = B1^2,
                                B2.2 = B2^2,
                                B3.2 = B3^2,
                                B4.2 = B4^2,
                                B5.2 = B5^2,
                                B7.2 = B7^2,
                                B2.B1 = B2/B1,
                                B3.B1 = B3/B1,
                                B4.B1 = B4/B1,
                                B5.B1 = B5/B1,
                                B7.B1 = B7/B1,
                                B3.B2 = B3/B2,
                                B4.B2 = B4/B2,
                                B5.B2 = B5/B2,
                                B7.B2 = B7/B2,
                                B4.B3 = B4/B3,
                                B5.B3 = B5/B3,
                                B7.B3 = B7/B3,
                                B5.B4 = B5/B4,
                                B7.B4 = B7/B4,
                                B7.B5 = B7/B5,
                                Latitude = lat,
                                Longitude = lon
                                # station_nm = paste0(0,station_no),
                                # site_no = paste0(0,site_no)
                              )][ 
                                # select only columns of interest
                                ,.(site_no, station_nm, year,
                                   # distance_km,
                                   # width, drainage_area_km2,
                                   Latitude,Longitude,sample_dt, num_pix, 
                                   snow_ice_qa_count,
                                   cloud_cover, cloud_qa_count,
                                   landsat_dt,
                                   B1,B2,B3,B4,B5,B6,B7,B2.B1,B3.B1,B4.B1,B5.B1,B7.B1,B3.B2,B4.B2,B5.B2,
                                   B7.B2,B4.B3,B5.B3,B7.B3,B5.B4,B7.B4,B7.B5, B1.2,B2.2,B3.2,B4.2,B5.2,B7.2
                                )][site_no != "0"][
                                  !((B6 < 2800 & B1 > 900 & B2 > 900 & B3 > 900 & B5 > 300 & B7 > 200 & B1 > B3 & B1 < B4) | # Elimate snowy & cold images
                                      (B1 > 700 & B1/B2 > 1.2 & B5 > 200)|
                                      ((B1 + B2 + B3 + B4) > 3200 & B3 < B1 & B3/B1 < 1.5 & B6 < 2750 & B5 > 300) |
                                      (B4 > 1500 & B4/B3 > 1.5 & B6 < 2800)| # This eliminates many cloudy/snowy images at high latitudes
                                      # ((B1 + B2 + B3 + B4) > 4000 & B6 < 2750 & B5 > 300 & abs(Latitude) > 40)
                                      ((B1 + B2 + B3 + B4) > 4000 & B6 < 2750 & B5 > 500 & abs(Latitude) > 40) # *changed B5 min to 500*
                                    # (B1 > 700 &
                                    # snow_ice_qa_count > (num_pix * 10) & 
                                    # snow_ice_qa_count > 500 &
                                    # B3/B1 < 1.5)
                                  )
                                ][
                                  ,':='(month = month(sample_dt),
                                        decade = ifelse(year < 1990, 1990,
                                                        ifelse(year > 2019, 2020,
                                                               year - year%%5)))]

# Remove pixels with snow, ice, clouds
river_import <- river_import[cloud_cover < 70 & !(num_pix < 2 & cloud_qa_count > 100)][
  yday(sample_dt) > 120 & yday(sample_dt) < 250 # Winter months
]

# Get arctic_thaw_slump River site numbers from USGS
arctic_thaw_slump_sites_all = river_import[
  # !(B3.B1 < 1 & B3 < 750)
][,.(Latitude = mean(Latitude, na.rm = T),
     Longitude = mean(Longitude, na.rm = T)),
  by = .(station_nm = gsub('st_', '', site_no))]

# Get site metadata
arctic_thaw_slump_site_metadata <- river_import[
  # !(B3.B1 < 1 & B3 < 750)
][,.(Latitude = mean(Latitude, na.rm = T),
     Longitude = mean(Longitude, na.rm = T),
     B3 = mean(B3, na.rm = T),
     B2 = mean(B2, na.rm = T),
     B1 = mean(B1, na.rm = T)),
  by = .(station_nm = gsub('st_', '', site_no), month)][month > 4 & month < 11]

arctic_thaw_slump_sites <- arctic_thaw_slump_sites_all

#### 2. CALCULATE SSC ####
# Apply SSC calibration models to make predictions based on new surface reflectance inputs (cluster needed)
# First, import function file
ssc_cluster_funs <- readRDS(paste0(wd_imports, 'SSC_cluster_function.rds'))

# And import cluster centers
clusters_calculated_list <- readRDS(paste0(wd_imports,'cluster_centers.rds'))
# Set number of cluster centers (6)
cluster_n_best <- 6
clustering_vars <- colnames(clusters_calculated_list[[cluster_n_best]]$centers)

# Scaling for cluster calculation
site_band_scaling <- readRDS(paste0(wd_imports,'site_band_scaling_all.rds'))
# Regressors
regressors_all <- c('B1', 'B2', 'B3', 'B4', 'B5', 'B7', # raw bands
                    'B1.2', 'B2.2', 'B3.2', 'B4.2', 'B5.2', 'B7.2', # squared bands
                    'site_no', # no clear way to add categorical variables
                    'B2.B1', 'B3.B1', 'B4.B1', 'B5.B1', 'B7.B1', # band ratios
                    'B3.B2', 'B4.B2', 'B5.B2', 'B7.B2',
                    'B4.B3', 'B5.B3', 'B7.B3',
                    'B5.B4', 'B7.B4', 'B7.B5')

# For base, cluster, and site predictions
getSSC_pred <- function(lm_data, regressors, cluster_funs){ # Version that includes site specification
  lm_data$pred_st <- NA
  lm_data[,ssc_subset:=cluster_sel] # clusters
  subsets <- unique(lm_data$ssc_subset)
  for(i in subsets){ # for individual cluster models
    # print(i)
    regressors_sel <- regressors[-which(regressors == 'site_no')]
    lm_data_lm <- lm_data[ssc_subset == i] # only chooses sites within cluster
    
    ssc_lm <- cluster_funs[[i]]
    glm_pred <- predict(ssc_lm, newx = as.matrix(lm_data_lm[,..regressors_sel]), s = "lambda.1se")
    lm_data[ssc_subset == i, pred_cl:= glm_pred]
    lm_data_lm <- NA
    # lm_data$res_cl[which(lm_data$ssc_subset == i)] <- resid(ssc_lm)
  }
  return(lm_data)
}

# Calculate cluster based on cluster function including scaling

# Cluster reflectance values by year, find breakpoint, cluster based on pre/post?
# OR do cluster by deforestation?
getCluster_monthly_decadal <- function(df,clustering_vars,n_centers, kmeans_object){
  # Compute band median at each site for clustering variables
  site_band_quantiles_all <- df[
    # n_insitu_samples_bySite][N_insitu_samples > 12
    ,.(N_samples = .N,
       B1 = median(B1),
       B2 = median(B2),
       B3 = median(B3),
       B4 = median(B4),
       # B5 = median(B5),
       # B7 = median(B7),
       B2.B1 = median(B2.B1),
       B3.B1 = median(B3.B1),
       B4.B1 = median(B4.B1),
       B3.B2 = median(B3.B2),
       B4.B2 = median(B4.B2),
       B4.B3 = median(B4.B3),
       B4.B3.B1 = median(B4.B3/B1)), 
    # by = .(station_nm,site_no, month, decade)]
    by = .(station_nm,site_no)]
  
  site_band_quantile_scaled <- scale(site_band_quantiles_all[,..clustering_vars], 
                                     center = attributes(site_band_scaling)$`scaled:center`[clustering_vars], 
                                     scale = attributes(site_band_scaling)$`scaled:scale`[clustering_vars])
  
  closest.cluster <- function(x) {
    cluster.dist <- apply(kmeans_object$centers, 1, function(y) sqrt(sum((x-y)^2)))
    return(which.min(cluster.dist)[1])
  }
  site_band_quantiles_all$cluster <- apply(site_band_quantile_scaled, 1, closest.cluster)
  
  df_cluster <- merge(df,
                      # site_band_quantiles_all[,c('site_no','station_nm','cluster', 'month','decade')], 
                      # by = c('site_no', 'station_nm','month','decade'))
                      site_band_quantiles_all[,c('site_no','station_nm','cluster')], 
                      by = c('site_no', 'station_nm'))
  df_cluster$cluster_sel <- df_cluster$cluster
  return(df_cluster)
  
}
# Get cluster for each site based on typical spectral profile
# This takes a long time to run
river_landsat_cl <- getCluster_monthly_decadal(river_import, 
                                               clustering_vars,cluster_n_best, 
                                               clusters_calculated_list[[cluster_n_best]])

# Run SSC prediction algorithm to get clustered prediction for SSC
river_landsat_pred <- getSSC_pred(na.omit(river_landsat_cl, cols = c(regressors_all, 'cluster_sel')), 
                                  regressors_all, ssc_cluster_funs)[,':='(
                                    SSC_mgL = ifelse(pred_cl > 5.5, NA, 10^pred_cl),
                                    month = month(sample_dt),
                                    decade = ifelse(year(sample_dt) < 1990, 1990,
                                                    ifelse(year(sample_dt) > 2019, 2015, 
                                                           year(sample_dt) - year(sample_dt)%%5)))]


#### 3. CLEAN DATA AND WRITE TO DRIVE ####
# Select just simple columns for export
river_landsat_pred_clean <- river_landsat_pred[
  ,.(site_no, station_nm, month, year, decade, Latitude, Longitude,sample_dt,
     num_pix, B1 = round(B1), B2 = round(B2), B3 = round(B3), B4 = round(B4), B6 = round(B6),
     cluster, SSC_mgL
  )
]

# Write full table to drive
fwrite(river_landsat_pred_clean, paste0(wd_imports,'arctic_thaw_slump_river_landsat_pred.csv'))

# Remove winter months from dataset
river_landsat_pred_clean_2 <- river_landsat_pred_clean[
  yday(sample_dt) > 80 & yday(sample_dt) < 260 # Winter months
]

# Add an additional filter for high SSC (mostly due to errors/artifacts)
river_landsat_pred_clean_3 <- river_landsat_pred_clean_2[SSC_mgL > 0.5 & SSC_mgL < 8000 &
                                                           !(SSC_mgL > 1000 & (B1 + B2 + B3) < 700)] 

# Write clean data to drive
fwrite(river_landsat_pred_clean_3, file = paste0(wd_imports, 'arctic_thaw_slump_river_ssc_warm_months.csv'))

#### 4A. CALCULATE AND PLOT WEEKLY SUMMARIES ####
# TO DO: ADD SITE-SPECIFIC REFERENCE YEAR AS COLUMN BY MERGING TABLE WITH REFERENCE
# Summarize SSC by site, bi-weekly
# Add a column, `reference` for whether slump has occurred
# Summarize SSC, Avg. and N samples for each 10-day period 
biweekly_SSC_by_site <- river_landsat_pred_clean_3[
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T),
     N_samples = .N),  
  by = .(site_no, station_nm, Latitude, Longitude, ten_day = yday(sample_dt)-yday(sample_dt)%%10, year)
][,':='(reference = factor(ifelse(grepl('_ref',site_no), 'Reference',
                                  ifelse(year > 2008, 'Affected, Post-slump', 'Affected, Pre-slump')),
                           levels = c('Reference', 'Affected, Pre-slump', 'Affected, Post-slump')))]

# Summarize number of samples, by week, by site
biweekly_SSC_sample_summary <- biweekly_SSC_by_site[
  ,.(N_samples = sum(N_samples, na.rm = T)),
  by = .(site_no = gsub('_ref', '', site_no), reference, ten_day)
]


river_landsat_pred_clean_3 <- river_landsat_pred_clean_3[
  ,':='(ten_day = yday(sample_dt)-yday(sample_dt)%%10,
        reference = factor(ifelse(grepl('_ref',site_no), 'Reference',
                                  ifelse(year > 1997, 'Affected, Post-slump', 'Affected, Pre-slump')),
                           levels = c('Reference', 'Affected, Pre-slump', 'Affected, Post-slump')))]

site_no_sel <- 's002'

# Select ten-day summarized SSC data
biweekly_SSC_by_site_sel <- biweekly_SSC_by_site[grepl(site_no_sel,site_no)]
biweekly_SSC_sample_summary_sel <- biweekly_SSC_sample_summary[grepl(site_no_sel,site_no)]

fwrite(river_landsat_pred_clean_3[grepl(site_no_sel,site_no)], 
       paste0(wd_imports, 'landsat_ssc_slump_', site_no_sel, '.csv'))

ssc_by_yday_pre_post_slump_s002 <- river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600
][
  !grepl('_ref',site_no)
][,':='(fractional_day = year + yday(sample_dt)/365,
        fractional_2month = year + (month-month%%2)/12)]


fwrite(ssc_by_yday_pre_post_slump_s002, paste0(wd_imports, '_', site_no_sel, 'ssc_by_yday_pre_post.csv'))

# STATISTIC: NUMBER OF LANDSAT IMAGES
# Affected
river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600
][
  !grepl('_ref',site_no)
][,uniqueN(sample_dt)]
# Reference
river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600
][
  grepl('_ref',site_no)
][,uniqueN(sample_dt)]


# FIGURE: Figure X. SSC timeseries for site s002, pre- and post-slump colored.
SSC_timeseries_by_site_plot <- ggplot(river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600
][
  # !grepl('_ref',site_no)
], 
aes(x = year + yday(sample_dt)/365, y = SSC_mgL,
    color = reference
)) + 
  stat_summary(aes(x = year + (month-month%%2)/12), geom = 'errorbar', width = 0.2, color = 'grey30') +
  geom_point(alpha = 0.3) +
  stat_summary(aes(x = year + (month-month%%2)/12), geom = 'line') + 
  stat_summary(aes(x = year + (month-month%%2)/12), geom = 'point') + 
  facet_wrap(gsub('_ref', '', site_no)~
               factor(ifelse(grepl('_ref',site_no), 'Reference', 'Affected'),
                      levels = c('Reference', 'Affected')),
             # scales = 'free_y', ncol = 2) +
             ncol = 1) +
  scale_color_manual(values = c('Reference' = '#41A8BF', 'Affected, Pre-slump' = 'steelblue', 'Affected, Post-slump' = 'goldenrod')) +
  season_facet +
  theme(legend.position = 'top') + 
  labs(
    x = 'Day of Year',
    y = 'SSC (mg/L)',
    color = ''
  )

ggsave(SSC_timeseries_by_site_plot, filename = paste0(wd_figures, region_name, '_SSC_timeseries_by_site_plot.png'),
       width = 8, height = 4.5)
ggsave(SSC_timeseries_by_site_plot, filename = paste0(wd_figures, region_name, '_SSC_timeseries_by_site_plot.pdf'),
       width = 8, height = 4.5, useDingbats = F)

# FIGURE: Figure X. Time distribution of landsat images
SSC_timeseries_time_distribution <- ggplot(river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600
][
  !grepl('_ref',site_no)
][
  ,':='(reference_simple = ifelse(grepl('ref', site_no), 'Reference', 'Affected'))
], 
aes(x = ten_day, fill = reference_simple)
) + 
  geom_bar(stat = 'count') +
  scale_fill_manual(values = c('Reference' = 'steelblue', 'Affected' = 'goldenrod')) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  facet_wrap(.~'Landsat') +
  season_facet +
  theme(legend.position = 'none') + 
  labs(
    x = 'Day of Year',
    y = 'Number of images',
    fill = ''
  )

ggsave(SSC_timeseries_time_distribution, filename = paste0(wd_figures, region_name, '_SSC_timeseries_time_distribution.png'),
       width = 8, height = 4.5)
ggsave(SSC_timeseries_time_distribution, filename = paste0(wd_figures, region_name, '_SSC_timeseries_time_distribution.pdf'),
       width = 8, height = 4.5, useDingbats = F)

SSC_timeseries_by_site_with_ref_plot <- ggplot(river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600
][
  ,':='(reference_simple = ifelse(grepl('ref', site_no), 'Reference', 'Affected'))
], 
aes(x = year + yday(sample_dt)/365, y = SSC_mgL,
    color = reference_simple
)) + 
  # stat_summary(aes(x = year), geom = 'errorbar', width = 0.2, color = 'grey30') +
  # geom_point(alpha = 0.3) +
  stat_summary(aes(x = year)) +
  stat_summary(aes(x = year), geom = 'line', fun = 'mean') + 
  geom_smooth(method = 'loess', span = 1, se = F) +
  scale_color_manual(values = c('Reference' = 'steelblue', 'Affected' = 'goldenrod')) +
  season_facet +
  theme(legend.position = 'top') + 
  labs(
    x = '',
    y = 'SSC (mg/L)',
    color = ''
  )

ggsave(SSC_timeseries_by_site_with_ref_plot, filename = paste0(wd_figures, region_name, '_SSC_timeseries_withRef_by_site_plot.png'),
       width = 8, height = 4.5)
ggsave(SSC_timeseries_by_site_with_ref_plot, filename = paste0(wd_figures, region_name, '_SSC_timeseries_withRef_by_site_plot.pdf'),
       width = 8, height = 4.5, useDingbats = F)

# 10-day period boxplots, control vs affected
biweekly_SSC_by_site_boxplot <- ggplot(river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][reference != 'Affected, Pre-slump'], 
                                       aes(x = factor(ten_day), y = SSC_mgL,
                                           fill = reference
                                           # fill = ifelse(year > 1999, 'Post-slump', 'Pre-slump'),
                                       )) + 
  geom_boxplot(outlier.shape = NA, notch = F) + 
  geom_text(data = biweekly_SSC_sample_summary_sel, 
            aes(y = max(biweekly_SSC_by_site_sel$SSC_mgL), label = N_samples),
            position=position_dodge(width = 0.7), angle = 90, hjust = 0, vjust = 0, size = 3.5) +
  # stat_summary(geom = 'line', fun = 'sum', aes(x = factor(week), group = reference)) +
  season_facet +
  theme(legend.position = 'top') + 
  facet_wrap(.~paste0('Site: ', gsub('_ref', '', site_no))) +
  # scale_y_continuous(limits = c(0, 350)) +
  scale_fill_manual(values = c('Reference' = '#41A8BF', 'Affected, Post-slump' = '#FF9D45')) +
  # scale_fill_manual(values = c('Control' = '#41A8BF', 'Affected, Pre-slump' = '#B0D1D9', 'Affected, Post-slump' = '#FF9D45')) +
  labs(
    x = 'Day of year',
    y = 'SSC (mg/L)',
    fill = ''
  )


ggsave(biweekly_SSC_by_site_boxplot, filename = paste0(wd_figures, region_name, '_', site_no_sel, '_biweekly_SSC_by_site_plot.png'),
       width = 6.5, height = 3.5)
ggsave(biweekly_SSC_by_site_boxplot, filename = paste0(wd_figures, region_name, '_', site_no_sel, '_biweekly_SSC_by_site_plot.pdf'),
       width = 6.5, height = 3.5)


#### 4B. ANNUAL AVERAGE SSC PLOTS ####

# Calculate annual average SSC using monthly averages 
river_landsat_pred_clean_annual_summary <- river_landsat_pred_clean_3[
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T)),
  by = .(site_no, Latitude, Longitude, station_nm, month, year)
][
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T)),
  by = .(site_no, station_nm, Latitude, Longitude, year)
]

# Write annual summary to drive
fwrite(river_landsat_pred_clean_annual_summary, file = paste0(wd_imports, region_name, '_ssc_warm_months_summary.csv'))

site_nos <- c('st_s002','st_s008')
# Plot annual timeseries for each site
annual_SSC_by_site_plot <- ggplot(river_landsat_pred_clean_3[
  site_no %in% site_nos
],
aes(x = year, y = SSC_mgL, group = site_no, 
    color = ifelse(grepl('ref', site_no), 'Reference', 'Affected'))) + 
  stat_summary(geom = 'errorbar', width = 0.2, color = 'grey30') +
  stat_summary(geom = 'line', fun = mean) + 
  scale_color_manual(values = c('Affected' = 'orange', 'Reference' = 'navy')) +
  facet_wrap(.~gsub("_ref", "", site_no), 
             scales = 'free_y', ncol = 1) +
  scale_x_continuous(labels = abbrev_year) +
  season_facet +
  labs(
    x = 'Year',
    y = 'SSC (mg/L)',
    color = 'Prediction model'
  )

ggsave(annual_SSC_by_site_plot, filename = paste0(wd_figures, region_name, '_SSC_by_site_plot.png'),
       width = 6.5, height = 9)
ggsave(annual_SSC_by_site_plot, filename = paste0(wd_figures, region_name, '_SSC_by_site_plot.pdf'),
       width = 6.5, height = 9, useDingbats = F)


#### 5. DISCHARGE DATA FROM WATER SURVEY OF CANADA ####
# Discharge data from tidyhydat
# Import all Canada discharge stations
ca_Q_stations <- data.table(hy_stations())

# Get s002 coordinates
s002_coords <- data.table('latitude' = 68.632051, 'longitude' = -131.756282)

s002_lat <- s002_coords[,latitude]
s002_long <- s002_coords[,longitude]

# Search Canadian stations for nearby discharge stations
ca_Q_stations <- ca_Q_stations[,':='(distance_to_miner_r = sqrt((LATITUDE - s002_lat)^2 + (LONGITUDE - s002_long)))]
# Find nearest 10 stations
ca_Q_s002_nearest <- ca_Q_stations[order(distance_to_miner_r)][1:10]

# Select travaillant river, get data and add water year and other metadata
travaillant_river_daily <- data.table(hy_daily_flows("10LB005"))[
  ,':='(water_year = ifelse(month(Date) > 9, year(Date) + 1, year(Date)),
        yday = yday(Date),
        yday10 = yday(Date) - yday(Date)%%10
  )
]



ggplot(travaillant_river_daily, aes(x = yday10, y = Value*(1548/1245))) +
  stat_summary()

# Summarize discharge data by water year
travaillant_river_Q_summary <- travaillant_river_daily[,.(Q_cms = mean(Value,na.rm = T)),
                                                       by = .(STATION_NUMBER, water_year)]

# Water year length
travaillant_river_open_water <- travaillant_river_daily[is.na(Symbol) & Value > 2 & yday > 100][
  ,.(ice_out = min(yday, na.rm = T),
     ice_in = max(yday, na.rm = T)),
  by = .(year = year(Date))
]

# Open water
travaillant_river_open_water_summary <- travaillant_river_open_water[
  ,.(ice_out = mean(ice_out, na.rm = T),
     ice_in = mean(ice_in, na.rm = T)
  )
]
# Import Daymet V4 precipitation data for travaillant and miner rivers
travaillant_river_precip <- fread(paste0(wd_dod_data, 'TRAVAILLANT_10LB005_Canada_precip.csv'))[
  ,':='(.geo = NULL)
]
miner_river_precip <- fread(paste0(wd_imports, 'slump_arctic_dem_dods/miner_river_watershed_precipitation_daymetv4.csv'))[
  ,':='(.geo = NULL)
]

# Add water year etc. columns to precip data
travaillant_river_precip <- travaillant_river_precip[,':='(
  water_year = ifelse(month(date) > 9, year(date) + 1, year(date)),
  yday = yday(date),
  yday10 = yday(date) - yday(date)%%10
)]

miner_river_precip <- miner_river_precip[,':='(
  water_year = ifelse(month(date) > 9, year(date) + 1, year(date)),
  yday = yday(date),
  yday10 = yday(date) - yday(date)%%10
)]


# Summarize precip data by water year
travaillant_river_precip_summary <- travaillant_river_precip[,.(precip_mm_yr = sum(mean,na.rm = T)),
                                                             by = .(site_no, water_year)]

miner_river_precip_summary <- miner_river_precip[,.(precip_mm_yr = sum(mean,na.rm = T)),
                                                 by = .(site_no, water_year)]

precip_miner_and_analogue <- rbind(miner_river_precip_summary[,':='(river = 'Miner River')], 
                                   travaillant_river_precip_summary[,':='(river = 'Travaillant River')], 
                                   use.names = T)

# Plot miner vs. travaillant river
ggplot(precip_miner_and_analogue, aes(x = water_year, y = precip_mm_yr)) + 
  geom_line(aes(group = river, color = river)) +
  theme_dark_mode +
  labs(
    x = 'Water year', 
    y = 'Annual Precipitation (mm)'
  )

# Combine precip and discharge data for Traivallant River (drainage area: 1245 km2)
travaillant_river_summary <- merge(
  travaillant_river_Q_summary,
  travaillant_river_precip_summary,
  on = 'water_year'
)[,':='(drainage_area_km2 = 1245)][
  ,':='(runoff_mm_yr = Q_cms * 3600 * 24 * 365.25/(drainage_area_km2*1000^2) * 1000)
]
# Add runoff ratio
travaillant_river_summary <- travaillant_river_summary[,':='(runoff_ratio = runoff_mm_yr/precip_mm_yr)]

# Summarize runofrealtime_plot
travaillant_river_runoff_ratio <- travaillant_river_summary[
  ,.(runoff_ratio = mean(runoff_ratio, na.rm = T),
     runoff_ratio_sd = sd(runoff_ratio, na.rm = T),
     runoff_ratio_se = sd(runoff_ratio, na.rm = T)/sqrt(.N),
     N_years = .N)
]

# Estimate Miner River discharge, 2013-2016
miner_river_precip_summary_2013_2017 <- miner_river_precip_summary[
  ,.(precip_mm_yr = mean(precip_mm_yr, na.rm = T),
     rel_error_Q = (sd(precip_mm_yr, na.rm = T))/mean(precip_mm_yr, na.rm = T))
][
  ,':='(runoff_mm_yr = precip_mm_yr * travaillant_river_runoff_ratio[,runoff_ratio])
]

# Calculate relative error using standard deviation
rel_error_Q = miner_river_precip_summary_2013_2017[,rel_error_Q]

# Plot Travaillant River during period of record
ggplot(travaillant_river_daily[
  # year(Date) >= 2013
], aes(x = Date, y = Value)) + 
  geom_line(color = 'white') + 
  theme_dark_mode

ggplot(travaillant_river_precip[year(date) >= 2013], aes(x = yday10, y = mean)) + 
  stat_summary(color = 'white') + 
  theme_dark_mode

ggplot(travaillant_river_summary, aes(x = water_year, y = precip_mm_yr)) + 
  geom_line(color = 'white') + 
  geom_line(aes(y = runoff_mm_yr), color = 'blue') +
  theme_dark_mode

ggplot(travaillant_river_summary, aes(x = precip_mm_yr, y = runoff_mm_yr)) + 
  geom_point(color = 'white') +
  theme_dark_mode


#### 6. ESTIMATE MASS LOSS FROM SLUMP SITE ####
# Density estimated for Canadia Arctic sites: https://doi.org/10.1029/2024GL108622 
permafrost_bulk_density <- mean(c(1.6,2.1,1.9,1.9,1.7,1.7,2.1,1.9))

ground_ice_fraction <- mean(25,12,30,50,50,63,30,30)/100
# Calculate erosion from s002 site
s002_example_erosion <- data.table(site_no = 's002', 
                                   erosion_m = 2.21, # 2017-06-17 - 2013-06-20
                                   n_years = 4,
                                   rmse = 0.729,
                                   slump_area_m2 = 75369,
                                   label = '-111531710744003',
                                   drainage_area_km2 = 1548, 
                                   runoff_mm_yr = miner_river_precip_summary_2013_2017[,runoff_mm_yr], # Calculated based on adjacent watershed
                                   flow_days = travaillant_river_open_water_summary[,ice_in] - travaillant_river_open_water_summary[,ice_out] # day of year without snow
)[,':='(
  erosion_m_yr = erosion_m/n_years
)][,':='(
  runoff_m3_yr = drainage_area_km2 * 1000^2 * runoff_mm_yr/1000,
  erosion_m3 = erosion_m * slump_area_m2,
  erosion_m3_yr = erosion_m_yr * slump_area_m2,
  erosion_tons_yr = erosion_m_yr * slump_area_m2*(1-ground_ice_fraction) * permafrost_bulk_density 
)][,':='(
  Q_cms = runoff_m3_yr/(flow_days * 3600 * 24),
  rel_error = rmse/n_years/erosion_m_yr, # rel error from DEM offset
  rel_error_Q = rel_error_Q, # Rel. error from Q record
  erosion_tons_s = erosion_tons_yr/(flow_days * 3600 * 24)
)][,':='(
  SSC_mgL = (erosion_tons_s * 1e9)/(Q_cms * 1000),
  SSC_mgL_error = (erosion_tons_s * 1e9)/(Q_cms * 1000)*sqrt(rel_error^2 + rel_error_Q^2),
  slump_erosion_error = erosion_m3_yr*rel_error,
  watershed_erosion_rate_mm_yr = erosion_m3_yr*(1-ground_ice_fraction)/(drainage_area_km2*1000^2)*1000,
  watershed_erosion_rate_mm_yr_error = erosion_m3_yr*(1-ground_ice_fraction)/(drainage_area_km2*1000^2)*1000*sqrt(rel_error^2 + rel_error_Q^2)
)]

s002_landsat_2013_2017_summary <- river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600 & year %in% c(2013:2016)
][
  ,':='(reference_simple = ifelse(grepl('ref', site_no), 'Reference', 'Affected'))
][
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T)),
  by = .(reference_simple, yday10 = ten_day, year)
][
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T)),
  by = .(reference_simple, year)
][
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T),
     SSC_mgL_rel_err = (sd(SSC_mgL, na.rm = T)/sqrt(.N))/mean(SSC_mgL, na.rm = T)),
  by = .(reference_simple)
]

s002_landsat_2013_2017_yday10_summary <- river_landsat_pred_clean_3[grepl(site_no_sel,site_no)][
  SSC_mgL < 600 & year %in% c(2013:2016)
][
  ,':='(reference_simple = ifelse(grepl('ref', site_no), 'Reference', 'Affected'))
][
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T)),
  by = .(reference_simple, year, yday10 = ten_day)
][
  ,.(SSC_mgL = mean(SSC_mgL, na.rm = T)),
  by = .(reference_simple, yday10)
]

# Combine timeseries from travaillant river and SSC record from landsat for 2013-2017
travaillant_river_yday10 = merge(
  travaillant_river_daily[
    ,.(Q_cms = mean(Value,na.rm = T)),
    by = .(STATION_NUMBER, yday10)
  ][,':='(
    Q_fraction = Q_cms/sum(Q_cms,na.rm = T)
  )],
  s002_landsat_2013_2017_yday10_summary,
  on = 'yday10'
)[,
  ':='(erosion_tons_yday10 = SSC_mgL * Q_cms * (3600*24*1000*10)/1000^3,
       erosion_tons_yday10_model = ifelse(reference_simple == 'Affected', 
                                          (SSC_mgL*2) * Q_cms * (3600*24*1000*10)/1000^3,
                                          SSC_mgL * Q_cms * (3600*24*1000*10)/1000^3)
  )]


monthly_sediment_flux <- merge(
  travaillant_river_yday10[,.(
    erosion_tons_yr = sum(erosion_tons_yday10, na.rm = T),
    erosion_tons_yday10_model = sum(erosion_tons_yday10_model, na.rm = T),
    SSC_mgL = mean(SSC_mgL, na.rm = T)),
    by = .(STATION_NUMBER, reference_simple)],
  s002_landsat_2013_2017_summary[,.(reference_simple, SSC_mgL_rel_err)],
  on = 'reference_simple')[
    ,':='(erosion_tons_yr_rel_error = sqrt(SSC_mgL_rel_err^2 + 0.71^2))
  ][
    ,':='(erosion_tons_yr_error = paste0('+/- ', round(erosion_tons_yr * erosion_tons_yr_rel_error,0), ' tons/yr'))
  ]
