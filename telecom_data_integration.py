#!/usr/bin/env python3
"""
Telecom Data Integration Script
Integrates TeleQnA and IPOD datasets to enhance telecom domain knowledge
Removes NGIL-specific data and prepares enhanced datasets for model retraining
"""

import pandas as pd
import numpy as np
import json
import re
import os
from typing import List, Dict, Set, Tuple
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelecomDataIntegrator:
    """Integrates telecom datasets for enhanced model training"""
    
    def __init__(self):
        self.telecom_skills = self._load_telecom_skills()
        self.telecom_job_titles = self._load_telecom_job_titles()
        
    def _load_telecom_skills(self) -> Set[str]:
        """Load comprehensive telecom skills from TeleQnA dataset"""
        telecom_skills = {
            # Core Telecom Technologies
            '5g', '4g', 'lte', '3g', '2g', 'gsm', 'cdma', 'wcdma', 'umts',
            'volte', 'vowifi', 'ims', 'sip', 'rtp', 'sdp', 'diameter',
            'ss7', 'sigtrans', 'm3ua', 'sctp', 'tcp', 'udp', 'ip', 'ipv6',
            
            # Network Technologies
            'ethernet', 'wifi', 'bluetooth', 'zigbee', 'lora', 'nb-iot',
            'cat-m1', 'cat-m2', 'cellular', 'mobile', 'wireless', 'optical',
            'fiber', 'copper', 'dsl', 'cable', 'satellite', 'microwave',
            
            # Telecom Standards & Protocols
            '3gpp', 'ieee', 'itu-t', 'etsi', 'ietf', 'rfc', 'itu-r',
            'gsma', 'oma', 'oneapi', 'parlay', 'camel', 'map', 'cap',
            'win', 'inap', 'isup', 'bicc', 'h.248', 'h.323', 'mgcp',
            
            # Network Management
            'snmp', 'netconf', 'yang', 'restconf', 'soap', 'xml', 'json',
            'corba', 'jms', 'mqtt', 'amqp', 'kafka', 'grpc', 'protobuf',
            
            # Telecom Services
            'sms', 'mms', 'ussd', 'cbs', 'lbs', 'roaming', 'handover',
            'mobility', 'authentication', 'authorization', 'billing',
            'charging', 'mediation', 'provisioning', 'activation',
            
            # Security & Privacy
            'encryption', 'authentication', 'authorization', 'privacy',
            'gdpr', 'pii', 'pdp', 'pcf', 'udm', 'udr', 'ausf', 'seaf',
            'tls', 'ssl', 'vpn', 'firewall', 'ids', 'ips', 'siem',
            
            # Cloud & Virtualization
            'nfv', 'sdn', 'openflow', 'openstack', 'kubernetes', 'docker',
            'vmware', 'hypervisor', 'container', 'microservices', 'api',
            'orchestration', 'automation', 'ci/cd', 'devops', 'gitops',
            
            # Data & Analytics
            'big data', 'hadoop', 'spark', 'kafka', 'elasticsearch',
            'mongodb', 'redis', 'postgresql', 'mysql', 'oracle', 'db2',
            'data warehouse', 'data lake', 'etl', 'elt', 'bi', 'analytics',
            
            # AI/ML in Telecom
            'machine learning', 'deep learning', 'neural networks', 'nlp',
            'computer vision', 'reinforcement learning', 'tensorflow',
            'pytorch', 'scikit-learn', 'xgboost', 'random forest', 'svm',
            'clustering', 'classification', 'regression', 'anomaly detection',
            
            # Telecom Equipment & Vendors
            'ericsson', 'nokia', 'huawei', 'zte', 'samsung', 'cisco',
            'juniper', 'alcatel-lucent', 'motorola', 'qualcomm', 'intel',
            'amd', 'arm', 'x86', 'risc', 'asic', 'fpga', 'dsp',
            
            # Testing & Quality
            'qos', 'qoe', 'kpi', 'sla', 'slo', 'monitoring', 'testing',
            'automation', 'robot framework', 'selenium', 'jmeter', 'loadrunner',
            'performance testing', 'stress testing', 'regression testing',
            
            # Project Management
            'agile', 'scrum', 'kanban', 'waterfall', 'prince2', 'pmp',
            'itil', 'cobit', 'iso', 'six sigma', 'lean', 'tqm',
            
            # Soft Skills
            'communication', 'leadership', 'teamwork', 'problem solving',
            'analytical thinking', 'project management', 'stakeholder management',
            'vendor management', 'contract negotiation', 'risk management'
        }
        return telecom_skills
    
    def _load_telecom_job_titles(self) -> Set[str]:
        """Load telecom-specific job titles from IPOD dataset"""
        telecom_job_titles = {
            'telecommunications engineer', 'network engineer', 'systems engineer',
            'software engineer', 'data engineer', 'devops engineer', 'site engineer',
            'field engineer', 'test engineer', 'quality engineer', 'reliability engineer',
            'network architect', 'solution architect', 'system architect', 'data architect',
            'enterprise architect', 'technical architect', 'infrastructure architect',
            'network administrator', 'system administrator', 'database administrator',
            'security administrator', 'cloud administrator', 'it administrator',
            'network analyst', 'data analyst', 'business analyst', 'systems analyst',
            'security analyst', 'performance analyst', 'quality analyst',
            'network manager', 'it manager', 'project manager', 'program manager',
            'product manager', 'service manager', 'operations manager',
            'network specialist', 'security specialist', 'data specialist',
            'telecom specialist', 'technical specialist', 'support specialist',
            'network consultant', 'it consultant', 'security consultant',
            'telecom consultant', 'technical consultant', 'business consultant',
            'network technician', 'field technician', 'support technician',
            'telecom technician', 'it technician', 'maintenance technician',
            'network operator', 'system operator', 'data center operator',
            'network coordinator', 'project coordinator', 'service coordinator',
            'network supervisor', 'team lead', 'technical lead', 'project lead',
            'network director', 'it director', 'technical director',
            'cto', 'cio', 'ciso', 'ctio', 'vp engineering', 'vp technology',
            'head of network', 'head of it', 'head of security',
            'chief network officer', 'chief technology officer', 'chief information officer'
        }
        return telecom_job_titles
    
    def process_teleqna_dataset(self, teleqna_file: str) -> pd.DataFrame:
        """Process TeleQnA dataset to extract telecom knowledge"""
        logger.info("Processing TeleQnA dataset...")
        
        try:
            with open(teleqna_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = []
            
            for question_id, question_data in data.items():
                # Extract question text
                question_text = question_data.get('question', '')
                category = question_data.get('category', '')
                explanation = question_data.get('explanation', '')
                
                # Extract telecom skills from question and explanation
                telecom_skills_found = []
                for skill in self.telecom_skills:
                    if skill.lower() in question_text.lower() or skill.lower() in explanation.lower():
                        telecom_skills_found.append(skill)
                
                if telecom_skills_found:
                    processed_data.append({
                        'text': question_text,
                        'skills': telecom_skills_found,
                        'category': category,
                        'explanation': explanation,
                        'source': 'TeleQnA'
                    })
            
            df = pd.DataFrame(processed_data)
            logger.info(f"Processed {len(df)} telecom-relevant questions from TeleQnA")
            return df
            
        except Exception as e:
            logger.error(f"Error processing TeleQnA dataset: {e}")
            return pd.DataFrame()
    
    def process_ipod_dataset(self, ipod_file: str) -> pd.DataFrame:
        """Process IPOD dataset to extract telecom job titles and skills"""
        logger.info("Processing IPOD dataset...")
        
        try:
            df = pd.read_csv(ipod_file)
            
            # Filter for telecom-related job titles
            telecom_jobs = []
            
            for _, row in df.iterrows():
                title = str(row.get('Original_Title', '')).lower()
                
                # Check if job title contains telecom-related keywords
                telecom_keywords = ['telecom', 'telecommunications', 'network', 'wireless', 
                                  'mobile', 'cellular', 'communication', '5g', '4g', 'lte']
                
                if any(keyword in title for keyword in telecom_keywords):
                    # Extract skills from job title
                    skills_found = []
                    for skill in self.telecom_skills:
                        if skill.lower() in title:
                            skills_found.append(skill)
                    
                    if skills_found:
                        telecom_jobs.append({
                            'job_title': row.get('Original_Title', ''),
                            'processed_title': row.get('Processed_Title', ''),
                            'skills': skills_found,
                            'tag_a1': row.get('Tag_A1', ''),
                            'tag_a2': row.get('Tag_A2', ''),
                            'tag_a3': row.get('Tag_A3', ''),
                            'source': 'IPOD'
                        })
            
            result_df = pd.DataFrame(telecom_jobs)
            logger.info(f"Processed {len(result_df)} telecom job titles from IPOD")
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing IPOD dataset: {e}")
            return pd.DataFrame()
    
    def process_gazetteer(self, gazetteer_file: str) -> pd.DataFrame:
        """Process gazetteer for telecom job titles and skills"""
        logger.info("Processing gazetteer dataset...")
        
        try:
            df = pd.read_csv(gazetteer_file)
            
            telecom_entries = []
            
            for _, row in df.iterrows():
                title = str(row.get('Title', '')).lower()
                
                # Check for telecom-related terms
                telecom_keywords = ['telecom', 'network', 'wireless', 'mobile', 'communication',
                                  'engineer', 'architect', 'analyst', 'manager', 'specialist']
                
                if any(keyword in title for keyword in telecom_keywords):
                    skills_found = []
                    for skill in self.telecom_skills:
                        if skill.lower() in title:
                            skills_found.append(skill)
                    
                    if skills_found:
                        telecom_entries.append({
                            'title': row.get('Title', ''),
                            'skills': skills_found,
                            'tag_a1': row.get('A1', ''),
                            'tag_a2': row.get('A2', ''),
                            'tag_a3': row.get('A3', ''),
                            'source': 'Gazetteer'
                        })
            
            result_df = pd.DataFrame(telecom_entries)
            logger.info(f"Processed {len(result_df)} telecom entries from gazetteer")
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing gazetteer: {e}")
            return pd.DataFrame()
    
    def remove_ngil_data(self, original_datasets: List[str]) -> List[pd.DataFrame]:
        """Remove NGIL-specific data from original datasets"""
        logger.info("Removing NGIL-specific data from original datasets...")
        
        cleaned_datasets = []
        
        for dataset_path in original_datasets:
            if os.path.exists(dataset_path):
                try:
                    df = pd.read_csv(dataset_path)
                    
                    # Remove rows containing NGIL-specific content
                    ngil_keywords = ['ngil', 'NGIL', 'ngil', 'NGIL']
                    
                    # Create mask to filter out NGIL content
                    mask = ~df.astype(str).apply(lambda x: x.str.contains('|'.join(ngil_keywords), case=False, na=False)).any(axis=1)
                    
                    cleaned_df = df[mask].copy()
                    
                    logger.info(f"Removed NGIL data from {dataset_path}: {len(df) - len(cleaned_df)} rows removed")
                    cleaned_datasets.append(cleaned_df)
                    
                except Exception as e:
                    logger.error(f"Error processing {dataset_path}: {e}")
        
        return cleaned_datasets
    
    def create_enhanced_training_data(self, output_dir: str = "enhanced_data"):
        """Create enhanced training datasets with telecom focus"""
        logger.info("Creating enhanced training datasets...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process telecom datasets
        teleqna_df = self.process_teleqna_dataset("TeleQnA.txt")
        ipod_df = self.process_ipod_dataset("NER_corpus_Collated_Positions_All_18June2020.csv")
        gazetteer_df = self.process_gazetteer("gazetteer.csv")
        
        # Remove NGIL data from original datasets
        original_datasets = [
            "AI_Resume_Screening.csv",
            "UpdatedResumeDataSet.csv", 
            "job_title_des.csv",
            "processed_job_data.csv"
        ]
        
        cleaned_datasets = self.remove_ngil_data(original_datasets)
        
        # Combine telecom data
        telecom_combined = pd.concat([
            teleqna_df,
            ipod_df,
            gazetteer_df
        ], ignore_index=True)
        
        # Save enhanced datasets
        telecom_combined.to_csv(f"{output_dir}/telecom_enhanced_data.csv", index=False)
        
        # Create skills-focused dataset
        skills_data = []
        for _, row in telecom_combined.iterrows():
            skills = row.get('skills', [])
            if isinstance(skills, list):
                for skill in skills:
                    skills_data.append({
                        'text': row.get('text', row.get('job_title', '')),
                        'skill': skill,
                        'category': row.get('category', ''),
                        'source': row.get('source', '')
                    })
        
        skills_df = pd.DataFrame(skills_data)
        skills_df.to_csv(f"{output_dir}/telecom_skills_data.csv", index=False)
        
        # Save cleaned original datasets
        for i, df in enumerate(cleaned_datasets):
            if len(df) > 0:
                df.to_csv(f"{output_dir}/cleaned_{original_datasets[i]}", index=False)
        
        # Create summary report
        self._create_summary_report(output_dir, telecom_combined, skills_df, cleaned_datasets)
        
        logger.info(f"Enhanced datasets saved to {output_dir}")
        return output_dir
    
    def _create_summary_report(self, output_dir: str, telecom_data: pd.DataFrame, 
                              skills_data: pd.DataFrame, cleaned_datasets: List[pd.DataFrame]):
        """Create a summary report of the enhanced datasets"""
        
        report = f"""
# Telecom Data Integration Summary Report

## Dataset Statistics

### Telecom Enhanced Data
- Total records: {len(telecom_data)}
- Unique skills found: {len(set([skill for skills in telecom_data['skills'] if isinstance(skills, list) for skill in skills]))}
- Sources: {telecom_data['source'].value_counts().to_dict()}

### Skills Data
- Total skill instances: {len(skills_data)}
- Unique skills: {skills_data['skill'].nunique()}
- Top skills: {skills_data['skill'].value_counts().head(10).to_dict()}

### Cleaned Original Datasets
"""
        
        for i, df in enumerate(cleaned_datasets):
            if len(df) > 0:
                report += f"- {df.shape[0]} records in cleaned dataset {i+1}\n"
        
        report += f"""
## Telecom Skills Coverage
The enhanced dataset now includes comprehensive coverage of:
- 5G/4G/LTE technologies
- Network protocols and standards
- Telecom equipment and vendors
- Security and privacy
- Cloud and virtualization
- AI/ML in telecom
- Project management in telecom

## Next Steps
1. Use enhanced_data/telecom_enhanced_data.csv for model retraining
2. Use enhanced_data/telecom_skills_data.csv for skills extraction training
3. Use cleaned original datasets for general training
"""
        
        with open(f"{output_dir}/integration_summary.md", 'w') as f:
            f.write(report)

def main():
    """Main function to run the telecom data integration"""
    logger.info("Starting telecom data integration...")
    
    integrator = TelecomDataIntegrator()
    
    # Create enhanced datasets
    output_dir = integrator.create_enhanced_training_data()
    
    logger.info("Telecom data integration completed successfully!")
    logger.info(f"Enhanced datasets saved to: {output_dir}")

if __name__ == "__main__":
    main()
