#!/bin/bash
#
# by AC Goglio (CMCC)
# annachiara.goglio@cmcc.it
#
# Written: 29/01/2021
#
set -u
set -e
#set -x 
########################
echo "*********** Data extraction for punctual harmonic analysis *********"
HERE="/users_home/oda/med_dev/src_dev/VAA_w08_w10/"
WORKDIR="/work/oda/med_dev/Venezia_Acqua_Alta_2019/VAA_wind_dir_extr_all/"
FLAG="2022"
#mkdir $WORKDIR/AN/
#mkdir $WORKDIR/EAS4/
#mkdir $WORKDIR/EAS56/
#mkdir $WORKDIR/2022/

echo "##### ANALYSIS #######"
if [[ $FLAG == "AN" || $FLAG == "ALL" ]] ; then
JOB2LINK="pextrjob_oldTG_atm_AN.temp"
for INI2LINK in p_extr_atm_EAS5_AN_w08.ini p_extr_atm_EAS5_AN_w10.ini p_extr_atm_EAS6_AN_w08.ini p_extr_atm_EAS6_AN_w10.ini ; do


   echo "JOB2LINK=$JOB2LINK"
   echo "INI2LINK=$INI2LINK"
   ln -svf $HERE/$JOB2LINK $HERE/pextrjob_oldTG.temp
   ln -svf $HERE/$INI2LINK $HERE/p_extr.ini

   PEXTR_INIFILE='p_extr.ini'
   SRC_DIR=$(pwd)
   
   # Check and load ini file
   if [[ -e ./${PEXTR_INIFILE} ]]; then
      echo "Reading ini file ${PEXTR_INIFILE}.."
      source ./${PEXTR_INIFILE}
      ANA_WORKDIR="$WORKDIR" #/AN/"
      echo "..Done"
   else
      echo "${PEXTR_INIFILE} file NOT FOUND in $(pwd)! Why?"
      exit
   fi
   
   if [[ ${TG_DATASET_TYPE} == "website" ]]; then
      JOB_TEMPLATE='pextrjob_oldTG.temp'
   else
      JOB_TEMPLATE='pextrjob_newTG.temp'
   fi
   echo "JOB_TEMPLATE=$JOB_TEMPLATE"
   
   JOB_TORUN='pextr.job'
   
   # Check job template file
   if [[ -e  ./${JOB_TEMPLATE} ]]; then
      echo "Found the job template ${JOB_TEMPLATE}"
   else
      echo "${JOB_TEMPLATE} file NOT FOUND in $(pwd)! Why?"
      exit 
   fi
   
   # Check work directory
   if [[ -d ${ANA_WORKDIR} ]]; then
      echo "Work dir: ${ANA_WORKDIR}"
      cp ${PEXTR_INIFILE} ${ANA_WORKDIR}/
   else
      echo "Work dir: ${ANA_WORKDIR} NOT FOUND! Why?"
      exit
   fi
   
   # Built the job from the template
   echo "I am building the job.."
   # Sed file creation and sobstitution of parematers in the templates  
   SED_FILE=sed_file.txt
   cat << EOF > ${ANA_WORKDIR}/${SED_FILE}
      s/%J_NAME%/${J_NAME//\//\\/}/g
      s/%J_OUT%/${J_OUT//\//\\/}/g
      s/%J_ERR%/${J_ERR//\//\\/}/g
      s/%J_QUEUE%/${J_QUEUE//\//\\/}/g
      s/%J_CPUS%/${J_CPUS//\//\\/}/g
      s/%J_PROJ%/${J_PROJ//\//\\/}/g
      #
      s/%SRC_DIR%/${SRC_DIR//\//\\/}/g
EOF
   
         sed -f ${ANA_WORKDIR}/${SED_FILE} ${JOB_TEMPLATE} > ${ANA_WORKDIR}/${JOB_TORUN}
         rm ${ANA_WORKDIR}/${SED_FILE}
   echo ".. Done"
   echo "Job path/name: ${ANA_WORKDIR}/${JOB_TORUN}"
   
   # Run the job
   echo "Submitting job ${J_NAME} to queue ${J_QUEUE} (Good luck!).."
   #bsub -P ${J_PROJ}<${ANA_WORKDIR}/${JOB_TORUN}
   echo "Check the output in ${ANA_WORKDIR} and/or the errors in ${J_ERR}!"
   #sleep 4m
done
fi

echo "##### EAS4 #######"
if [[ $FLAG == "EAS4" || $FLAG == "ALL" ]]; then
JOB2LINK_AR=("pextrjob_oldTG_atm_FC1_EAS4.temp" "pextrjob_oldTG_atm_FC2_EAS4.temp" "pextrjob_oldTG_atm_FC3_EAS4.temp")
INI2LINK_AR=("p_extr_atm_EAS4_FC1_w08.ini" "p_extr_atm_EAS4_FC2_w08.ini" "p_extr_atm_EAS4_FC3_w08.ini")
for IDX2LINK in 0 1 2  ; do

   JOB2LINK=${JOB2LINK_AR[$IDX2LINK]}
   INI2LINK=${INI2LINK_AR[$IDX2LINK]}

   echo "JOB2LINK=$JOB2LINK"
   echo "INI2LINK=$INI2LINK"
   ln -svf $HERE/$JOB2LINK $HERE/pextrjob_oldTG.temp
   ln -svf $HERE/$INI2LINK $HERE/p_extr.ini

   PEXTR_INIFILE='p_extr.ini'
   SRC_DIR=$(pwd)

   # Check and load ini file
   if [[ -e ./${PEXTR_INIFILE} ]]; then
      echo "Reading ini file ${PEXTR_INIFILE}.."
      source ./${PEXTR_INIFILE}
      ANA_WORKDIR="$WORKDIR" #/EAS4/"
      echo "..Done"
   else
      echo "${PEXTR_INIFILE} file NOT FOUND in $(pwd)! Why?"
      exit
   fi

   if [[ ${TG_DATASET_TYPE} == "website" ]]; then
      JOB_TEMPLATE='pextrjob_oldTG.temp'
   else
      JOB_TEMPLATE='pextrjob_newTG.temp'
   fi
   echo "JOB_TEMPLATE=$JOB_TEMPLATE"

   JOB_TORUN='pextr.job'

   # Check job template file
   if [[ -e  ./${JOB_TEMPLATE} ]]; then
      echo "Found the job template ${JOB_TEMPLATE}"
   else
      echo "${JOB_TEMPLATE} file NOT FOUND in $(pwd)! Why?"
      exit
   fi

   # Check work directory
   if [[ -d ${ANA_WORKDIR} ]]; then
      echo "Work dir: ${ANA_WORKDIR}"
      cp ${PEXTR_INIFILE} ${ANA_WORKDIR}/
   else
      echo "Work dir: ${ANA_WORKDIR} NOT FOUND! Why?"
      exit
   fi

   # Built the job from the template
   echo "I am building the job.."
   # Sed file creation and sobstitution of parematers in the templates  
   SED_FILE=sed_file.txt
   cat << EOF > ${ANA_WORKDIR}/${SED_FILE}
      s/%J_NAME%/${J_NAME//\//\\/}/g
      s/%J_OUT%/${J_OUT//\//\\/}/g
      s/%J_ERR%/${J_ERR//\//\\/}/g
      s/%J_QUEUE%/${J_QUEUE//\//\\/}/g
      s/%J_CPUS%/${J_CPUS//\//\\/}/g
      s/%J_PROJ%/${J_PROJ//\//\\/}/g
      #
      s/%SRC_DIR%/${SRC_DIR//\//\\/}/g
EOF

         sed -f ${ANA_WORKDIR}/${SED_FILE} ${JOB_TEMPLATE} > ${ANA_WORKDIR}/${JOB_TORUN}
         rm ${ANA_WORKDIR}/${SED_FILE}
   echo ".. Done"
   echo "Job path/name: ${ANA_WORKDIR}/${JOB_TORUN}"

   # Run the job
   echo "Submitting job ${J_NAME} to queue ${J_QUEUE} (Good luck!).."
   bsub -P ${J_PROJ}<${ANA_WORKDIR}/${JOB_TORUN}
   echo "Check the output in ${ANA_WORKDIR} and/or the errors in ${J_ERR}!"
   sleep 4m
done
fi

echo "##### EAS56 #######"
if [[ $FLAG == "EAS56" || $FLAG == "ALL" ]]; then
for JOB2LINK in "pextrjob_oldTG_atm_FC1_EAS56_w10.temp" "pextrjob_oldTG_atm_FC1_EAS56.temp" "pextrjob_oldTG_atm_FC2_EAS56.temp" "pextrjob_oldTG_atm_FC3_EAS56.temp" ; do
   if [[  $JOB2LINK == "pextrjob_oldTG_atm_FC1_EAS56_w10.temp" ]]; then
      INI2LINK_AR=("p_extr_atm_EAS56_FC1_w10.ini")
   elif [[  $JOB2LINK == "pextrjob_oldTG_atm_FC1_EAS56.temp" ]]; then
      INI2LINK_AR=("p_extr_atm_EAS56_FC1_w08.ini")
   elif [[  $JOB2LINK == "pextrjob_oldTG_atm_FC2_EAS56.temp" ]]; then
      INI2LINK_AR=("p_extr_atm_EAS56_FC2_w10.ini" "p_extr_atm_EAS56_FC2_w08.ini")
   elif [[  $JOB2LINK == "pextrjob_oldTG_atm_FC3_EAS56.temp" ]]; then
      INI2LINK_AR=("p_extr_atm_EAS56_FC3_w10.ini" "p_extr_atm_EAS56_FC3_w08.ini")
   fi

   for IDX2LINK in ${INI2LINK_AR[@]} ; do
      echo "JOB2LINK=$JOB2LINK"
      echo "INI2LINK=$IDX2LINK"
      ln -svf $HERE/$JOB2LINK $HERE/pextrjob_oldTG.temp
      ln -svf $HERE/$IDX2LINK $HERE/p_extr.ini

      PEXTR_INIFILE='p_extr.ini'
      SRC_DIR=$(pwd)

      # Check and load ini file
      if [[ -e ./${PEXTR_INIFILE} ]]; then
         echo "Reading ini file ${PEXTR_INIFILE}.."
         source ./${PEXTR_INIFILE}
         ANA_WORKDIR="$WORKDIR" #/EAS56/"
         echo "..Done"
      else
         echo "${PEXTR_INIFILE} file NOT FOUND in $(pwd)! Why?"
         exit
      fi
   
      if [[ ${TG_DATASET_TYPE} == "website" ]]; then
         JOB_TEMPLATE='pextrjob_oldTG.temp'
      else
         JOB_TEMPLATE='pextrjob_newTG.temp'
      fi
      echo "JOB_TEMPLATE=$JOB_TEMPLATE"
   
      JOB_TORUN='pextr.job'
   
      # Check job template file
      if [[ -e  ./${JOB_TEMPLATE} ]]; then
         echo "Found the job template ${JOB_TEMPLATE}"
      else
         echo "${JOB_TEMPLATE} file NOT FOUND in $(pwd)! Why?"
         exit
      fi
   
      # Check work directory
      if [[ -d ${ANA_WORKDIR} ]]; then
         echo "Work dir: ${ANA_WORKDIR}"
         cp ${PEXTR_INIFILE} ${ANA_WORKDIR}/
      else
         echo "Work dir: ${ANA_WORKDIR} NOT FOUND! Why?"
         exit
      fi
   
      # Built the job from the template
      echo "I am building the job.."
      # Sed file creation and sobstitution of parematers in the templates  
      SED_FILE=sed_file.txt
      cat << EOF > ${ANA_WORKDIR}/${SED_FILE}
         s/%J_NAME%/${J_NAME//\//\\/}/g
         s/%J_OUT%/${J_OUT//\//\\/}/g
         s/%J_ERR%/${J_ERR//\//\\/}/g
         s/%J_QUEUE%/${J_QUEUE//\//\\/}/g
         s/%J_CPUS%/${J_CPUS//\//\\/}/g
         s/%J_PROJ%/${J_PROJ//\//\\/}/g
         #
         s/%SRC_DIR%/${SRC_DIR//\//\\/}/g
EOF
   
      sed -f ${ANA_WORKDIR}/${SED_FILE} ${JOB_TEMPLATE} > ${ANA_WORKDIR}/${JOB_TORUN}
      rm ${ANA_WORKDIR}/${SED_FILE}
      echo ".. Done"
      echo "Job path/name: ${ANA_WORKDIR}/${JOB_TORUN}"
   
      # Run the job
      echo "Submitting job ${J_NAME} to queue ${J_QUEUE} (Good luck!).."
      bsub -P ${J_PROJ}<${ANA_WORKDIR}/${JOB_TORUN}
      echo "Check the output in ${ANA_WORKDIR} and/or the errors in ${J_ERR}!"
      sleep 4m 
   done
done
fi 

echo "##### 2022 #######"
if [[ $FLAG == "2022" || $FLAG == "ALL" ]]; then
JOB2LINK_AR=("pextrjob_oldTG_atm_FC1_EAS56_2022.temp" "pextrjob_oldTG_atm_FC2_EAS56_2022.temp" "pextrjob_oldTG_atm_FC3_EAS56_2022.temp")
INI2LINK_AR=("p_extr_atm_EAS56_FC1_w10_2022.ini" "p_extr_atm_EAS56_FC2_w10_2022.ini" "p_extr_atm_EAS56_FC3_w10_2022.ini")
INI2LINK_AR=("p_extr_wind_EAS56_FC1_w10_2022.ini" "p_extr_wind_EAS56_FC2_w10_2022.ini" "p_extr_wind_EAS56_FC3_w10_2022.ini")
for IDX2LINK in 0 1 2 ; do

   JOB2LINK=${JOB2LINK_AR[$IDX2LINK]}
   INI2LINK=${INI2LINK_AR[$IDX2LINK]}

   echo "JOB2LINK=$JOB2LINK"
   echo "INI2LINK=$INI2LINK"
   ln -svf $HERE/$JOB2LINK $HERE/pextrjob_oldTG.temp
   ln -svf $HERE/$INI2LINK $HERE/p_extr.ini

   PEXTR_INIFILE='p_extr.ini'
   SRC_DIR=$(pwd)

   # Check and load ini file
   if [[ -e ./${PEXTR_INIFILE} ]]; then
      echo "Reading ini file ${PEXTR_INIFILE}.."
      source ./${PEXTR_INIFILE}
      ANA_WORKDIR="$WORKDIR" #/2022/"
      echo "..Done"
   else
      echo "${PEXTR_INIFILE} file NOT FOUND in $(pwd)! Why?"
      exit
   fi

   if [[ ${TG_DATASET_TYPE} == "website" ]]; then
      JOB_TEMPLATE='pextrjob_oldTG.temp'
   else
      JOB_TEMPLATE='pextrjob_newTG.temp'
   fi
   echo "JOB_TEMPLATE=$JOB_TEMPLATE"

   JOB_TORUN='pextr.job'

   # Check job template file
   if [[ -e  ./${JOB_TEMPLATE} ]]; then
      echo "Found the job template ${JOB_TEMPLATE}"
   else
      echo "${JOB_TEMPLATE} file NOT FOUND in $(pwd)! Why?"
      exit
   fi

   # Check work directory
   if [[ -d ${ANA_WORKDIR} ]]; then
      echo "Work dir: ${ANA_WORKDIR}"
      cp ${PEXTR_INIFILE} ${ANA_WORKDIR}/
   else
      echo "Work dir: ${ANA_WORKDIR} NOT FOUND! Why?"
      exit
   fi

   # Built the job from the template
   echo "I am building the job.."
   # Sed file creation and sobstitution of parematers in the templates  
   SED_FILE=sed_file.txt
   cat << EOF > ${ANA_WORKDIR}/${SED_FILE}
      s/%J_NAME%/${J_NAME//\//\\/}/g
      s/%J_OUT%/${J_OUT//\//\\/}/g
      s/%J_ERR%/${J_ERR//\//\\/}/g
      s/%J_QUEUE%/${J_QUEUE//\//\\/}/g
      s/%J_CPUS%/${J_CPUS//\//\\/}/g
      s/%J_PROJ%/${J_PROJ//\//\\/}/g
      #
      s/%SRC_DIR%/${SRC_DIR//\//\\/}/g
EOF

         sed -f ${ANA_WORKDIR}/${SED_FILE} ${JOB_TEMPLATE} > ${ANA_WORKDIR}/${JOB_TORUN}
         rm ${ANA_WORKDIR}/${SED_FILE}
   echo ".. Done"
   echo "Job path/name: ${ANA_WORKDIR}/${JOB_TORUN}"

   # Run the job
   echo "Submitting job ${J_NAME} to queue ${J_QUEUE} (Good luck!).."
   bsub -P ${J_PROJ}<${ANA_WORKDIR}/${JOB_TORUN}
   echo "Check the output in ${ANA_WORKDIR} and/or the errors in ${J_ERR}!"
   sleep 4m
done
fi 
