library(tidyverse)
library(lme4)
library(lmerTest)
library(pROC)
library(dplyr)

setwd("F:\\git\\network_miner")
df_hfo = read.csv("F:\\manuscripts\\manuiscript_0001_hfo_rates\\ravi_hfo_numbers~N59+v03.csv")
df_sourcesink = read.csv("F:\\git\\network_miner\\sourceSink_Data_v01.csv")

df_sourcesink <- df_sourcesink %>%
  filter(pid %in% df_hfo$patient)

# good outcome = ILAE 1-3
dfHFO_ilae13 = df_hfo %>% filter(ilae<=3)
dfHFO_ilae_not13 = df_hfo %>% filter(ilae>=4)

dfSS_ilae13 = df_sourcesink %>% filter(ilae<=3)
dfSS_ilae_not13 = df_sourcesink %>% filter(ilae>=4)



mod_5_3 = glmer(soz~POP_znorm_log_rpm+(1|patient), family="binomial", data=dfHFO_ilae13); 

mod_sink_dist = glmer(soz~sink_dist_mats+(1|pid), family="binomial", data=dfSS_ilae13); 
mod_source_dist = glmer(soz~source_dist_mats+(1|pid), family="binomial", data=dfSS_ilae13); 
mod_source_infl = glmer(soz~source_infl_mats+(1|pid), family="binomial", data=dfSS_ilae13); 
mod_sink_conn = glmer(soz~sink_conn_mats+(1|pid), family="binomial", data=dfSS_ilae13); 
mod_product = glmer(soz~ss_ind_mats+(1|pid), family="binomial", data=dfSS_ilae13); 



# Generate predicted probabilities for each model
dfHFO_ilae13$prob_5_3 <- predict(mod_5_3, type="response")
dfSS_ilae13$prob_sink_dist <- predict(mod_sink_dist, type="response")
dfSS_ilae13$prob_source_dist <- predict(mod_source_dist, type="response")
dfSS_ilae13$prob_source_infl <- predict(mod_source_infl, type="response")
dfSS_ilae13$prob_sink_conn <- predict(mod_sink_conn, type="response")
dfSS_ilae13$prob_product <- predict(mod_product, type="response")

# Get ROC curves for each model
roc_5_3 <- roc(dfHFO_ilae13$soz, dfHFO_ilae13$prob_5_3)
roc_sink_dist <- roc(dfSS_ilae13$soz, dfSS_ilae13$prob_sink_dist)
roc_source_dist <- roc(dfSS_ilae13$soz, dfSS_ilae13$prob_source_dist)
roc_source_infl <- roc(dfSS_ilae13$soz, dfSS_ilae13$prob_source_infl)
roc_sink_conn <- roc(dfSS_ilae13$soz, dfSS_ilae13$prob_sink_conn)
roc_product <- roc(dfSS_ilae13$soz, dfSS_ilae13$prob_product)

# Combine ROC curve data
roc_data <- data.frame(
  sensitivity = c(roc_5_3$sensitivities, roc_sink_dist$sensitivities, roc_source_dist$sensitivities, 
                  roc_source_infl$sensitivities, roc_sink_conn$sensitivities, 
                  roc_product$sensitivities),
  specificity = c(1 - roc_5_3$specificities,1 - roc_sink_dist$specificities, 1 - roc_source_dist$specificities, 
                  1 - roc_source_infl$specificities, 1 - roc_sink_conn$specificities, 
                  1 - roc_product$specificities),
  model = rep(c("Model 5_3","Sink Distance", "Source Distance", "Source Influence", 
                "Sink Connectivity", "Product\n(sink dist * source influence * sink connectivity)"),
              times = c(length(roc_5_3$sensitivities),length(roc_sink_dist$sensitivities), length(roc_source_dist$sensitivities), 
                        length(roc_source_infl$sensitivities), length(roc_sink_conn$sensitivities),
                        length(roc_product$sensitivities)))
)



# Plot the ROC curves
ggplot(roc_data, aes(x = specificity, y = sensitivity, color = model)) +
  geom_line(linewidth=1) +
  theme_minimal() +
  labs(title = "ROC Curves for All Models", x = "1 - Specificity", y = "Sensitivity") +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom", 
        plot.background = element_rect(fill = "white", color = NA),   # Opaque white background
        panel.background = element_rect(fill = "white", color = NA)) # Opaque white panel background
ggsave("roc_curves_plot.png", width = 8, height = 6, bg = "white",dpi=300)

