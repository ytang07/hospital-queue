myopic <- read.csv("myopic.csv")
random <- read.csv("random.csv")
colnames(random) <- c("Expected Survivals", "Total Patients", "# Immediate Patients", "# Delayed Patients")
colnames(myopic) <- c("Expected Survivals", "Total Patients", "# Immediate Patients", "# Delayed Patients")
testres <- t.test(myopic[1], random[1], paired = "TRUE", alternative="greater")
testres$p.value
msurv <- myopic[,1]
rsurv <- random[,1]
mnum <- myopic[,2]
rnum <- random[,2]
perc_surv_m <- msurv/mnum
perc_surv_r <- rsurv/rnum
conf <- function(mean, sd, n){
   err <- qnorm(0.975)*sd/sqrt(n)
   low <- mean-err
   high <- mean+err
   list(low, high)
}
print(conf(mean(perc_surv_m), sd(perc_surv_m), 99))
print(conf(mean(perc_surv_r), sd(perc_surv_r), 99))