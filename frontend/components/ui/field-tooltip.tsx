"use client";

import React, { useState } from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface FieldTooltipProps {
  label: string;
  badge?: string;
  description: string;
  clinicalNote?: string;
}

export const heartFieldInfo = {
  age: {
    label: "Age (years)",
    badge: "18 – 100 yrs",
    description:
      "Chronological age of the patient in years. Coronary artery disease risk progressively escalates with age as arterial walls stiffen and cumulative exposure to risk factors increases.",
    clinicalNote:
      "Cardiovascular risk rises significantly in men aged 45+ and women aged 55+.",
  },
  sex: {
    label: "Gender",
    badge: "Biological Sex",
    description:
      "Biological sex of the patient. Males typically present with coronary heart disease at an earlier age, whereas female risk accelerates markedly post-menopause.",
    clinicalNote: "Used to calibrate baseline cardiovascular risk metrics.",
  },
  chest_pain_type: {
    label: "Chest Pain Type",
    badge: "4 Categories",
    description:
      "Clinical classification: Typical Angina (substernal discomfort provoked by exertion/stress, relieved by rest), Atypical Angina (meets some criteria), Non-anginal Pain (sharp or localized discomfort), or Asymptomatic (absence of symptoms).",
    clinicalNote:
      "Typical angina is a hallmark symptom of coronary artery disease.",
  },
  resting_bp: {
    label: "Resting Blood Pressure",
    badge: "Normal: < 120 mmHg",
    description:
      "Resting systolic blood pressure (in mm Hg) measured upon hospital admission. Chronic high pressure strains arterial walls and increases myocardial workload.",
    clinicalNote:
      "Readings 130–139 mm Hg indicate Stage 1 hypertension; >= 140 mm Hg indicate Stage 2.",
  },
  cholesterol: {
    label: "Serum Cholesterol",
    badge: "Normal: < 200 mg/dl",
    description:
      "Total serum cholesterol level in mg/dl. Elevated low-density lipoprotein levels promote atheromatous plaque formation, narrowing coronary arteries.",
    clinicalNote:
      "Desirable: < 200 mg/dl; Borderline: 200–239 mg/dl; High risk: >= 240 mg/dl.",
  },
  fasting_bs: {
    label: "Fasting Blood Sugar",
    badge: "Normal: <= 120 mg/dl",
    description:
      "Blood glucose concentration after an overnight fast (> 8 hours). Fasting sugar > 120 mg/dl indicates impaired fasting glycemia or diabetes mellitus.",
    clinicalNote:
      "Diabetic status significantly increases risk of silent myocardial ischemia.",
  },
  resting_ecg: {
    label: "Resting ECG",
    badge: "3 Categories",
    description:
      "Resting electrocardiogram findings: Normal (regular baseline rhythm), ST-T Wave Abnormality (T-wave inversions or ST shifts > 0.05 mV), or Left Ventricular Hypertrophy (enlarged left ventricular muscle mass).",
    clinicalNote:
      "LVH commonly develops secondary to chronic untreated hypertension.",
  },
  max_hr: {
    label: "Maximum Heart Rate",
    badge: "Target: ~220 - Age",
    description:
      "Peak heart rate (beats per minute) reached during graded exercise stress testing. Reduced peak heart rate (chronotropic incompetence) signals diminished cardiac reserve.",
    clinicalNote:
      "Failure to reach 85% of age-predicted maximum suggests cardiac impairment.",
  },
  exercise_angina: {
    label: "Exercise Induced Angina",
    badge: "No / Yes",
    description:
      "Presence of chest pain or constricting discomfort provoked during physical exertion. Points to a supply-demand mismatch in coronary myocardial perfusion.",
    clinicalNote:
      "A 'Yes' finding is a strong diagnostic predictor of obstructive coronary disease.",
  },
  oldpeak: {
    label: "Oldpeak (ST Depression)",
    badge: "Normal: < 1.0 mm",
    description:
      "Exercise-induced ST segment depression in millimeters relative to resting baseline ECG. Quantifies the severity and depth of myocardial ischemia under physical exertion.",
    clinicalNote:
      "Depression >= 1.0–2.0 mm reflects clinically significant subendocardial ischemia.",
  },
  st_slope: {
    label: "ST Slope",
    badge: "3 Profiles",
    description:
      "Slope angle of the peak exercise ST segment: Upsloping (typically benign physiological response), Flat (horizontal segment indicating ischemia), or Downsloping (marker of severe multivessel coronary disease).",
    clinicalNote:
      "Flat and downsloping slopes carry significant predictive weight for adverse cardiac events.",
  },
};

export function FieldTooltip({
  label,
  badge,
  description,
  clinicalNote,
}: FieldTooltipProps) {
  const [open, setOpen] = useState(false);

  return (
    <Tooltip open={open} onOpenChange={setOpen}>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            setOpen((prev) => !prev);
          }}
          className="cursor-pointer text-brand-fg/60 hover:text-brand-fg focus:text-brand-fg transition-colors inline-flex items-center justify-center p-0.5 rounded-full focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-fg/50"
          aria-label={`Clinical information for ${label}`}
        >
          <Info className="w-3.5 h-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        align="start"
        sideOffset={6}
        className="z-50 w-72 sm:w-80 rounded-xl border border-white/15 bg-neutral-950/95 p-3.5 text-xs text-neutral-100 shadow-2xl backdrop-blur-md"
      >
        <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-2">
          <span className="font-semibold text-xs tracking-tight text-white">
            {label}
          </span>
          {badge && (
            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30 whitespace-nowrap">
              {badge}
            </span>
          )}
        </div>
        <p className="pt-2 text-[11px] leading-relaxed text-neutral-300">
          {description}
        </p>
        {clinicalNote && (
          <div className="mt-2 rounded-lg bg-white/5 p-2 border border-white/5 text-[10px] text-neutral-400">
            <span className="font-semibold text-neutral-300">Clinical context: </span>
            {clinicalNote}
          </div>
        )}
      </TooltipContent>
    </Tooltip>
  );
}
