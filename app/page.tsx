"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Database,
  Search,
  Brain,
  Target,
  BarChart3,
  Cloud,
  CheckCircle,
  Circle,
  Award,
  Code,
  Zap,
  FileCode,
  GraduationCap,
  Play,
  Settings,
  Download,
  RefreshCw,
  TrendingUp,
  AlertTriangle,
  CheckSquare,
} from "lucide-react"

// Import Sidebar components
import {
  Sidebar,
  SidebarProvider,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarInset,
  SidebarTrigger,
} from "@/components/ui/sidebar"

function FraudDetectionProject() {
  const [currentStep, setCurrentStep] = useState(1)
  const [code, setCode] = useState("")
  const [output, setOutput] = useState("")
  const [showSolution, setShowSolution] = useState(false)

  const steps = [
    {
      id: 1,
      title: "Step 1: Load and Explore Dataset",
      description: "Load the credit card fraud dataset and examine class distribution",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load dataset
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

# Basic info
print("Dataset shape:", df.shape)
print("\\nFirst 5 rows:")
print(df.head())

# Check class distribution
print("\\nClass distribution:")
print(df['Class'].value_counts(normalize=True))

# Visualize class imbalance
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Class')
plt.title("Class Distribution (Before Balancing)")
plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
plt.ylabel("Count")
plt.show()`,
      solution: `# Expected Output:
# Dataset shape: (284807, 31)
# Class distribution:
# 0    0.998273  ← 99.83% Non-Fraud
# 1    0.001727  ← 0.17% Fraud
# 
# This shows severe class imbalance!`,
    },
    {
      id: 2,
      title: "Step 2: Data Preprocessing",
      description: "Handle missing values and scale features",
      code: `# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Scale the 'Amount' feature
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Drop original Amount column
df = df.drop(['Amount'], axis=1)

print("\\nDataset after preprocessing:")
print(df.head())`,
      solution: `# Expected Output:
# Missing values per column:
# All columns: 0 (no missing values)
# 
# Amount feature has been standardized`,
    },
    {
      id: 3,
      title: "Step 3: Train-Test Split",
      description: "Split data while maintaining class distribution",
      code: `from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("\\nTraining set class distribution:")
print(y_train.value_counts(normalize=True))`,
      solution: `# Expected Output:
# Training set shape: (227845, 30)
# Test set shape: (56962, 30)
# Class distribution maintained in both sets`,
    },
    {
      id: 4,
      title: "Step 4: Apply SMOTE for Balancing",
      description: "Use SMOTE to create synthetic minority samples",
      code: `from imblearn.over_sampling import SMOTE

# Check original distribution
print("Before SMOTE:")
print("Training set distribution:", Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\\nAfter SMOTE:")
print("Balanced training set distribution:", Counter(y_train_balanced))

# Visualize the difference
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
ax1.bar(['Non-Fraud', 'Fraud'], [Counter(y_train)[0], Counter(y_train)[1]])
ax1.set_title('Before SMOTE')
ax1.set_ylabel('Count')

# After SMOTE
ax2.bar(['Non-Fraud', 'Fraud'], [Counter(y_train_balanced)[0], Counter(y_train_balanced)[1]])
ax2.set_title('After SMOTE')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()`,
      solution: `# Expected Output:
# Before SMOTE: {0: 227451, 1: 394}
# After SMOTE: {0: 227451, 1: 227451}
# Perfect balance achieved!`,
    },
    {
      id: 5,
      title: "Step 5: Train Classification Models",
      description: "Train models with and without SMOTE for comparison",
      code: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Model 1: Without SMOTE
model_original = LogisticRegression(max_iter=1000, random_state=42)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

# Model 2: With SMOTE
model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_balanced, y_train_balanced)
y_pred_smote = model_smote.predict(X_test)

print("=== Model WITHOUT SMOTE ===")
print(classification_report(y_test, y_pred_original, 
                          target_names=['Non-Fraud', 'Fraud']))

print("\\n=== Model WITH SMOTE ===")
print(classification_report(y_test, y_pred_smote, 
                          target_names=['Non-Fraud', 'Fraud']))`,
      solution: `# Expected improvements with SMOTE:
# - Higher recall for fraud detection
# - Better F1-score for minority class
# - More balanced precision-recall trade-off`,
    },
    {
      id: 6,
      title: "Step 6: Evaluate and Compare Results",
      description: "Visualize confusion matrices and compare key metrics",
      code: `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\\n{model_name} Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return precision, recall, f1

# Calculate metrics for both models
metrics_original = calculate_metrics(y_test, y_pred_original, "WITHOUT SMOTE")
metrics_smote = calculate_metrics(y_test, y_pred_smote, "WITH SMOTE")

# Visualize confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix without SMOTE
cm1 = confusion_matrix(y_test, y_pred_original)
ConfusionMatrixDisplay(cm1, display_labels=['Non-Fraud', 'Fraud']).plot(ax=ax1)
ax1.set_title('Without SMOTE')

# Confusion matrix with SMOTE
cm2 = confusion_matrix(y_test, y_pred_smote)
ConfusionMatrixDisplay(cm2, display_labels=['Non-Fraud', 'Fraud']).plot(ax=ax2)
ax2.set_title('With SMOTE')

plt.tight_layout()
plt.show()

# Compare F1 scores
f1_comparison = [metrics_original[2], metrics_smote[2]]
plt.figure(figsize=(8, 6))
plt.bar(['Without SMOTE', 'With SMOTE'], f1_comparison, color=['red', 'green'])
plt.title('F1-Score Comparison')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
for i, v in enumerate(f1_comparison):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.show()`,
      solution: `# Key Insights:
# - SMOTE typically improves recall for fraud detection
# - May slightly reduce precision but improves overall F1-score
# - Better balance between catching frauds and avoiding false alarms`,
    },
  ]

  const currentStepData = steps.find((s) => s.id === currentStep) || steps[0]

  const runCode = () => {
    const outputs = {
      1: "Dataset shape: (284807, 31)\n\nClass distribution:\n0    0.998273\n1    0.001727\n\nSevere class imbalance detected!",
      2: "Missing values per column:\nAll columns: 0\n\nAmount feature successfully scaled.",
      3: "Training set shape: (227845, 30)\nTest set shape: (56962, 30)\n\nStratified split completed.",
      4: "Before SMOTE: {0: 227451, 1: 394}\nAfter SMOTE: {0: 227451, 1: 227451}\n\nPerfect balance achieved!",
      5: "Models trained successfully.\n\nWithout SMOTE - F1: 0.123\nWith SMOTE - F1: 0.456\n\nSMOTE shows significant improvement!",
      6: "Evaluation completed.\n\nSMOTE model shows:\n- Higher recall for fraud detection\n- Better balanced performance\n- Improved F1-score",
    }

    setOutput(outputs[currentStep] || "Code executed successfully!")
  }

  const downloadNotebook = () => {
    const notebookContent = {
      cells: steps.map((step, index) => ({
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [`# ${step.title}\n`, `# ${step.description}\n\n`, step.code],
      })),
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3",
        },
        language_info: {
          name: "python",
          version: "3.8.0",
        },
      },
      nbformat: 4,
      nbformat_minor: 4,
    }

    const blob = new Blob([JSON.stringify(notebookContent, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "fraud_detection_project.ipynb"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-red-50 to-orange-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🏠 Credit Card Fraud Detection Project</h3>
        <p className="text-gray-700 mb-4">
          Complete end-to-end project demonstrating how to handle severe class imbalance in fraud detection using SMOTE
          and proper evaluation metrics.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Class Imbalance</Badge>
          <Badge variant="outline">SMOTE</Badge>
          <Badge variant="outline">Fraud Detection</Badge>
          <Badge variant="outline">Real Dataset</Badge>
        </div>
        <div className="mt-4">
          <Button onClick={downloadNotebook} variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Download Jupyter Notebook
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {steps.map((step) => (
          <Button
            key={step.id}
            variant={currentStep === step.id ? "default" : "outline"}
            size="sm"
            onClick={() => setCurrentStep(step.id)}
          >
            Step {step.id}
          </Button>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">{currentStepData.title}</CardTitle>
          <CardDescription className="text-base">{currentStepData.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="code" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="code">Code</TabsTrigger>
              <TabsTrigger value="output">Output</TabsTrigger>
              <TabsTrigger value="explanation">Explanation</TabsTrigger>
            </TabsList>

            <TabsContent value="code" className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">Python Code</h4>
                <div className="flex gap-2">
                  <Button onClick={runCode} size="sm">
                    <Play className="h-4 w-4 mr-2" />
                    Run Code
                  </Button>
                  <Button onClick={() => setShowSolution(!showSolution)} variant="outline" size="sm">
                    {showSolution ? "Hide" : "Show"} Expected Output
                  </Button>
                </div>
              </div>

              <textarea
                value={currentStepData.code}
                readOnly
                className="w-full h-64 p-4 font-mono text-sm border rounded-lg bg-gray-900 text-green-400 resize-none"
              />

              {showSolution && (
                <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
                  <h5 className="font-semibold text-blue-900 mb-2">Expected Output:</h5>
                  <pre className="text-blue-800 text-sm whitespace-pre-wrap">{currentStepData.solution}</pre>
                </div>
              )}
            </TabsContent>

            <TabsContent value="output" className="space-y-4">
              <h4 className="font-semibold">Execution Output</h4>
              <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm min-h-32 whitespace-pre-wrap">
                {output || "Click 'Run Code' to see output..."}
              </div>
            </TabsContent>

            <TabsContent value="explanation" className="space-y-4">
              <h4 className="font-semibold">Step Explanation</h4>
              <div className="prose max-w-none">{getStepExplanation(currentStep)}</div>
            </TabsContent>
          </Tabs>

          <div className="flex justify-between mt-6">
            <Button
              variant="outline"
              onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
              disabled={currentStep === 1}
            >
              Previous Step
            </Button>
            <Button onClick={() => setCurrentStep(Math.min(6, currentStep + 1))} disabled={currentStep === 6}>
              Next Step
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function getStepExplanation(step: number) {
  const explanations = {
    1: (
      <div>
        <p>
          In this step, we load the credit card fraud dataset and examine its structure. The key insight is the severe
          class imbalance:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>99.83% of transactions are legitimate (Class 0)</li>
          <li>Only 0.17% are fraudulent (Class 1)</li>
        </ul>
        <p className="mt-2">
          This extreme imbalance will cause standard ML algorithms to achieve high accuracy by simply predicting "no
          fraud" for everything.
        </p>
      </div>
    ),
    2: (
      <div>
        <p>Data preprocessing is crucial for model performance:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Check for missing values (fortunately, this dataset has none)</li>
          <li>Scale the 'Amount' feature to prevent it from dominating other features</li>
          <li>The other features (V1-V28) are already scaled due to PCA transformation</li>
        </ul>
      </div>
    ),
    3: (
      <div>
        <p>Proper train-test splitting is essential:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Use stratified splitting to maintain class distribution in both sets</li>
          <li>This ensures our test set is representative of the original data</li>
          <li>80-20 split provides enough data for training while reserving sufficient data for testing</li>
        </ul>
      </div>
    ),
    4: (
      <div>
        <p>SMOTE (Synthetic Minority Oversampling Technique) addresses class imbalance by:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Creating synthetic examples of the minority class</li>
          <li>Interpolating between existing minority samples and their nearest neighbors</li>
          <li>Achieving perfect balance (50-50 split) in the training set</li>
          <li>Avoiding simple duplication which could lead to overfitting</li>
        </ul>
      </div>
    ),
    5: (
      <div>
        <p>Training two models allows us to compare the impact of SMOTE:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Model 1: Trained on original imbalanced data</li>
          <li>Model 2: Trained on SMOTE-balanced data</li>
          <li>Both use Logistic Regression for fair comparison</li>
          <li>The balanced model should show better minority class detection</li>
        </ul>
      </div>
    ),
    6: (
      <div>
        <p>Evaluation focuses on metrics appropriate for imbalanced classification:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>
            <strong>Precision:</strong> Of predicted frauds, how many are actually frauds?
          </li>
          <li>
            <strong>Recall:</strong> Of actual frauds, how many did we catch?
          </li>
          <li>
            <strong>F1-Score:</strong> Harmonic mean balancing precision and recall
          </li>
          <li>Confusion matrices show the trade-offs between false positives and false negatives</li>
        </ul>
      </div>
    ),
  }

  return explanations[step] || <div>Explanation not available</div>
}

function MissingDataDashboard() {
  const [strategy, setStrategy] = useState("mean")
  const [threshold, setThreshold] = useState([50])
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState(null)

  // Sample dataset with missing values
  const sampleData = {
    columns: ["age", "income", "education", "experience", "score"],
    missingCounts: [12, 8, 15, 3, 0],
    totalRows: 1000,
    missingPatterns: [
      { pattern: "age_missing", count: 12, percentage: 1.2 },
      { pattern: "income_missing", count: 8, percentage: 0.8 },
      { pattern: "education_missing", count: 15, percentage: 1.5 },
      { pattern: "experience_missing", count: 3, percentage: 0.3 },
    ],
  }

  const handleImputation = () => {
    setIsProcessing(true)

    // Simulate processing
    setTimeout(() => {
      const imputedResults = {
        strategy: strategy,
        columnsProcessed: sampleData.columns.filter((_, i) => sampleData.missingCounts[i] > 0),
        beforeMissing: sampleData.missingCounts.reduce((a, b) => a + b, 0),
        afterMissing: 0,
        imputedValues: {
          age: strategy === "mean" ? 35.2 : strategy === "median" ? 34.0 : "Adult",
          income: strategy === "mean" ? 52000 : strategy === "median" ? 48000 : "Medium",
          education: strategy === "mode" ? "Bachelor" : "Bachelor",
          experience: strategy === "mean" ? 8.5 : strategy === "median" ? 8.0 : "Mid-level",
        },
      }
      setResults(imputedResults)
      setIsProcessing(false)
    }, 2000)
  }

  const resetData = () => {
    setResults(null)
    setStrategy("mean")
    setThreshold([50])
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎛️ Missingness Map Dashboard</h3>
        <p className="text-gray-700 mb-4">
          Interactive tool to visualize and handle missing data patterns using sample employee dataset.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Missing Data</Badge>
          <Badge variant="outline">Visualization</Badge>
          <Badge variant="outline">Imputation</Badge>
          <Badge variant="outline">Sample Data</Badge>
        </div>
      </div>

      {/* Explanation Section */}
      <Card>
        <CardHeader>
          <CardTitle>How to Use This Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-gray-700">
              This interactive dashboard helps you understand and handle missing data patterns. Here's what you can do:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">📊 Visualize Missing Data</h4>
                <p className="text-blue-800 text-sm">
                  See which columns have missing values and their percentages. The color coding helps identify severity:
                  green (low), yellow (medium), red (high).
                </p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">🔧 Apply Imputation</h4>
                <p className="text-green-800 text-sm">
                  Choose different strategies (mean, median, mode) and see how they fill missing values. Set thresholds
                  to automatically drop columns with too many missing values.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Imputation Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Sample Dataset:</strong> Employee data with {sampleData.totalRows} rows and{" "}
                {sampleData.columns.length} columns
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Imputation Strategy</label>
              <Select value={strategy} onValueChange={setStrategy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mean">Mean (for numerical)</SelectItem>
                  <SelectItem value="median">Median (for numerical)</SelectItem>
                  <SelectItem value="mode">Mode (for categorical)</SelectItem>
                  <SelectItem value="knn">KNN Imputer</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium">Drop Threshold: {threshold[0]}%</label>
              <Slider value={threshold} onValueChange={setThreshold} max={100} min={0} step={5} className="mt-2" />
              <p className="text-xs text-gray-600 mt-1">
                Columns with more than {threshold[0]}% missing values will be dropped
              </p>
            </div>

            <div className="flex gap-2">
              <Button onClick={handleImputation} className="flex-1" disabled={isProcessing}>
                {isProcessing ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Settings className="h-4 w-4 mr-2" />
                    Apply Imputation
                  </>
                )}
              </Button>
              <Button onClick={resetData} variant="outline">
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Missing Data Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Missing data heatmap visualization */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-3">Missing Values by Column</h4>
                <div className="space-y-2">
                  {sampleData.columns.map((col, index) => {
                    const missingCount = sampleData.missingCounts[index]
                    const percentage = (missingCount / sampleData.totalRows) * 100
                    return (
                      <div key={col} className="flex items-center justify-between">
                        <span className="text-sm font-medium">{col}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                percentage > threshold[0]
                                  ? "bg-red-500"
                                  : percentage > 5
                                    ? "bg-yellow-500"
                                    : "bg-green-500"
                              }`}
                              style={{ width: `${Math.max(percentage, 2)}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-600 w-12">
                            {missingCount} ({percentage.toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Summary statistics */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <div className="font-medium text-blue-900">Total Missing</div>
                  <div className="text-blue-800">{sampleData.missingCounts.reduce((a, b) => a + b, 0)} values</div>
                </div>
                <div className="bg-green-50 p-3 rounded-lg">
                  <div className="font-medium text-green-900">Completeness</div>
                  <div className="text-green-800">
                    {(
                      100 -
                      (sampleData.missingCounts.reduce((a, b) => a + b, 0) /
                        (sampleData.totalRows * sampleData.columns.length)) *
                        100
                    ).toFixed(1)}
                    %
                  </div>
                </div>
              </div>

              {/* Results after imputation */}
              {results && (
                <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                  <h4 className="font-semibold text-green-900 mb-2">✅ Imputation Complete</h4>
                  <div className="space-y-2 text-sm">
                    <p className="text-green-800">
                      <strong>Strategy:</strong> {results.strategy}
                    </p>
                    <p className="text-green-800">
                      <strong>Missing values before:</strong> {results.beforeMissing}
                    </p>
                    <p className="text-green-800">
                      <strong>Missing values after:</strong> {results.afterMissing}
                    </p>
                    <div className="mt-3">
                      <p className="text-green-800 font-medium">Sample imputed values:</p>
                      <div className="grid grid-cols-2 gap-2 mt-1">
                        {Object.entries(results.imputedValues).map(([col, val]) => (
                          <div key={col} className="text-xs">
                            <span className="font-mono">{col}:</span> {val}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function ImbalanceDashboard() {
  const [technique, setTechnique] = useState("smote")
  const [ratio, setRatio] = useState([1.0])
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState(null)

  // Sample imbalanced dataset (fraud detection scenario)
  const originalData = {
    class0: 9850, // Non-fraud
    class1: 150, // Fraud
    total: 10000,
  }

  const handleResampling = () => {
    setIsProcessing(true)

    setTimeout(() => {
      let newClass0, newClass1

      switch (technique) {
        case "smote":
          // SMOTE creates synthetic minority samples
          newClass0 = originalData.class0
          newClass1 = Math.round(originalData.class0 * ratio[0])
          break
        case "random_over":
          // Random oversampling duplicates minority samples
          newClass0 = originalData.class0
          newClass1 = Math.round(originalData.class0 * ratio[0])
          break
        case "random_under":
          // Random undersampling reduces majority samples
          newClass1 = originalData.class1
          newClass0 = Math.round(originalData.class1 / ratio[0])
          break
        case "tomek":
          // Tomek links removes borderline samples
          newClass0 = Math.round(originalData.class0 * 0.95)
          newClass1 = originalData.class1
          break
        default:
          newClass0 = originalData.class0
          newClass1 = originalData.class1
      }

      setResults({
        technique,
        ratio: ratio[0],
        before: { class0: originalData.class0, class1: originalData.class1 },
        after: { class0: newClass0, class1: newClass1 },
        balanceRatio: newClass1 / newClass0,
        totalSamples: newClass0 + newClass1,
      })
      setIsProcessing(false)
    }, 1500)
  }

  const resetData = () => {
    setResults(null)
    setTechnique("smote")
    setRatio([1.0])
  }

  const getBalanceColor = (ratio) => {
    if (ratio > 0.8) return "text-green-600"
    if (ratio > 0.3) return "text-yellow-600"

    return "text-red-600"
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎛️ Imbalance Fixer Studio</h3>
        <p className="text-gray-700 mb-4">
          Interactive tool to balance imbalanced datasets using various resampling techniques on fraud detection data.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Class Imbalance</Badge>
          <Badge variant="outline">SMOTE</Badge>
          <Badge variant="outline">Resampling</Badge>
          <Badge variant="outline">Fraud Detection</Badge>
        </div>
      </div>

      {/* Explanation Section */}
      <Card>
        <CardHeader>
          <CardTitle>Understanding Class Imbalance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-gray-700">
              Class imbalance occurs when one class significantly outnumbers another. This dashboard demonstrates
              different techniques to address this problem:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-900 mb-2">🔄 Resampling Techniques</h4>
                <ul className="text-purple-800 text-sm space-y-1">
                  <li>
                    <strong>SMOTE:</strong> Creates synthetic minority samples
                  </li>
                  <li>
                    <strong>Random Over:</strong> Duplicates minority samples
                  </li>
                  <li>
                    <strong>Random Under:</strong> Removes majority samples
                  </li>
                  <li>
                    <strong>Tomek Links:</strong> Cleans borderline samples
                  </li>
                </ul>
              </div>
              <div className="bg-pink-50 p-4 rounded-lg">
                <h4 className="font-semibold text-pink-900 mb-2">📊 Visual Comparison</h4>
                <p className="text-pink-800 text-sm">
                  See how each technique affects the class distribution. The balance ratio shows how close you get to
                  equal representation (1.0 = perfect balance).
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Resampling Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-red-50 p-3 rounded-lg">
              <p className="text-sm text-red-800">
                <strong>Original Dataset:</strong> {originalData.total} samples
                <br />
                <strong>Imbalance:</strong> {((originalData.class1 / originalData.total) * 100).toFixed(1)}% fraud cases
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Resampling Technique</label>
              <Select value={technique} onValueChange={setTechnique}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="smote">SMOTE (Synthetic Oversampling)</SelectItem>
                  <SelectItem value="random_over">Random Oversampling</SelectItem>
                  <SelectItem value="random_under">Random Undersampling</SelectItem>
                  <SelectItem value="tomek">Tomek Links (Cleaning)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-gray-600 mt-1">
                {technique === "smote" && "Creates synthetic minority samples using interpolation"}
                {technique === "random_over" && "Duplicates existing minority samples randomly"}
                {technique === "random_under" && "Removes majority samples randomly"}
                {technique === "tomek" && "Removes borderline samples to clean decision boundary"}
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Target Ratio: {ratio[0]}</label>
              <Slider value={ratio} onValueChange={setRatio} max={1.0} min={0.1} step={0.1} className="mt-2" />
              <p className="text-xs text-gray-600 mt-1">
                {technique === "random_under"
                  ? "Minority to majority ratio after undersampling"
                  : "Minority to majority ratio after oversampling"}
              </p>
            </div>

            <div className="flex gap-2">
              <Button onClick={handleResampling} className="flex-1" disabled={isProcessing}>
                {isProcessing ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Apply Resampling
                  </>
                )}
              </Button>
              <Button onClick={resetData} variant="outline">
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Class Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-3 text-center">Before Resampling</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Non-Fraud</span>
                      <span className="text-sm font-mono">{originalData.class0}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className="bg-blue-500 h-4 rounded-full"
                        style={{ width: `${(originalData.class0 / originalData.total) * 100}%` }}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm">Fraud</span>
                      <span className="text-sm font-mono">{originalData.class1}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className="bg-red-500 h-4 rounded-full"
                        style={{ width: `${(originalData.class1 / originalData.total) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="mt-3 text-center">
                    <Badge variant="secondary" className="text-xs">
                      Ratio: {(originalData.class1 / originalData.class0).toFixed(3)}
                    </Badge>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-3 text-center">After Resampling</h4>
                  {results ? (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Non-Fraud</span>
                        <span className="text-sm font-mono">{results.after.class0}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-4">
                        <div
                          className="bg-blue-500 h-4 rounded-full"
                          style={{ width: `${(results.after.class0 / results.totalSamples) * 100}%` }}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-sm">Fraud</span>
                        <span className="text-sm font-mono">{results.after.class1}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-4">
                        <div
                          className="bg-red-500 h-4 rounded-full"
                          style={{ width: `${(results.after.class1 / results.totalSamples) * 100}%` }}
                        />
                      </div>

                      <div className="mt-3 text-center">
                        <Badge variant="secondary" className="text-xs">
                          Ratio: {results.balanceRatio.toFixed(3)}
                        </Badge>
                      </div>
                    </div>
                  ) : (
                    <div className="h-32 bg-gray-100 rounded-lg flex items-center justify-center">
                      <p className="text-gray-500 text-sm">Apply resampling to see results</p>
                    </div>
                  )}
                </div>
              </div>

              {results && (
                <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                  <h4 className="font-semibold text-green-900 mb-2">✅ Resampling Complete</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-green-800">
                        <strong>Technique:</strong> {results.technique.toUpperCase()}
                      </p>
                      <p className="text-green-800">
                        <strong>Total Samples:</strong> {results.totalSamples.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className={`font-medium ${getBalanceColor(results.balanceRatio)}`}>
                        <strong>Balance Score:</strong> {(results.balanceRatio * 100).toFixed(1)}%
                      </p>
                      <p className="text-green-800">
                        <strong>Change:</strong> {results.totalSamples > originalData.total ? "+" : ""}
                        {(((results.totalSamples - originalData.total) / originalData.total) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function FeatureSynthesisDashboard() {
  const [entityCount, setEntityCount] = useState(3)
  const [maxDepth, setMaxDepth] = useState([2])
  const [isGenerating, setIsGenerating] = useState(false)
  const [results, setResults] = useState(null)

  const sampleEntities = {
    customers: { rows: 1000, features: ["customer_id", "age", "income", "signup_date"] },
    transactions: { rows: 15000, features: ["transaction_id", "customer_id", "amount", "timestamp", "merchant_type"] },
    products: { rows: 500, features: ["product_id", "category", "price", "rating"] },
    reviews: { rows: 8000, features: ["review_id", "customer_id", "product_id", "rating", "review_date"] },
    sessions: { rows: 25000, features: ["session_id", "customer_id", "duration", "pages_viewed", "session_date"] },
  }

  const generateFeatures = () => {
    setIsGenerating(true)

    setTimeout(() => {
      const baseFeatures = 15
      const syntheticFeatures = Math.max(1, entityCount * maxDepth[0] * 3)

      const generatedFeatures = [
        "customer_avg_transaction_amount",
        "customer_total_transactions_last_30_days",
        "customer_unique_merchants_count",
        "customer_max_transaction_amount",
        "customer_transaction_frequency",
        "customer_days_since_last_transaction",
        "customer_transaction_amount_trend",
        "customer_weekend_transaction_ratio",
        "customer_avg_product_rating",
        "customer_favorite_category",
        "customer_review_sentiment_avg",
        ...(maxDepth[0] >= 2
          ? [
              "customer_avg_session_duration_per_transaction",
              "customer_product_diversity_score",
              "customer_merchant_loyalty_score",
              "customer_seasonal_spending_pattern",
            ]
          : []),
      ]

      setResults({
        totalFeatures: Math.max(baseFeatures + syntheticFeatures, generatedFeatures.length),
        baseFeatures,
        syntheticFeatures: Math.max(syntheticFeatures, generatedFeatures.length - baseFeatures),
        features: generatedFeatures.slice(0, 12),
        primitives: ["sum", "mean", "count", "max", "min", "std", "trend", "time_since"],
        relationships: [
          { from: "customers", to: "transactions", type: "one-to-many" },
          { from: "customers", to: "reviews", type: "one-to-many" },
          { from: "customers", to: "sessions", type: "one-to-many" },
          { from: "transactions", to: "products", type: "many-to-one" },
        ].slice(0, entityCount - 1),
      })
      setIsGenerating(false)
    }, 3000)
  }

  const resetGeneration = () => {
    setResults(null)
    setEntityCount(3)
    setMaxDepth([2])
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎛️ Feature Synthesizer</h3>
        <p className="text-gray-700 mb-4">
          Automatically generate features from relational e-commerce data using Deep Feature Synthesis (DFS).
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Featuretools</Badge>
          <Badge variant="outline">DFS</Badge>
          <Badge variant="outline">Auto Features</Badge>
          <Badge variant="outline">E-commerce Data</Badge>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>How Feature Synthesis Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-gray-700">
              This tool demonstrates automated feature engineering using Deep Feature Synthesis (DFS). It creates new
              features by combining data across related tables:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">🔗 Entity Relationships</h4>
                <p className="text-green-800 text-sm">
                  The system understands how tables relate (customers → transactions → products) and creates features
                  that span these relationships.
                </p>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">🧮 Automatic Calculations</h4>
                <p className="text-blue-800 text-sm">
                  Features are automatically calculated using primitives like SUM, MEAN, COUNT across different time
                  windows and entity relationships.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Entity Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Sample Dataset:</strong> E-commerce platform with customer transactions, reviews, and sessions
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Number of Entities: {entityCount}</label>
              <Slider
                value={[entityCount]}
                onValueChange={(value) => setEntityCount(value[0])}
                max={5}
                min={2}
                step={1}
                className="mt-2"
              />
              <div className="text-xs text-gray-600 mt-1">
                Active entities: {Object.keys(sampleEntities).slice(0, entityCount).join(", ")}
              </div>
            </div>

            <div>
              <label className="text-sm font-medium">Max Synthesis Depth: {maxDepth[0]}</label>
              <Slider value={maxDepth} onValueChange={setMaxDepth} max={3} min={1} step={1} className="mt-2" />
              <p className="text-xs text-gray-600 mt-1">
                {maxDepth[0] === 1 && "Basic aggregations (sum, mean, count)"}
                {maxDepth[0] === 2 && "Advanced features (trends, ratios, cross-entity)"}
                {maxDepth[0] === 3 && "Deep synthesis (complex multi-entity features)"}
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="font-medium text-sm">Entity Overview</h4>
              <div className="space-y-1">
                {Object.entries(sampleEntities)
                  .slice(0, entityCount)
                  .map(([name, info]) => (
                    <div key={name} className="flex justify-between text-xs bg-gray-50 p-2 rounded">
                      <span className="font-medium">{name}</span>
                      <span>
                        {info.rows.toLocaleString()} rows, {info.features.length} features
                      </span>
                    </div>
                  ))}
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={generateFeatures} className="flex-1" disabled={isGenerating}>
                {isGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-2" />
                    Generate Features
                  </>
                )}
              </Button>
              <Button onClick={resetGeneration} variant="outline">
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Feature Generation Results</CardTitle>
          </CardHeader>
          <CardContent>
            {results ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-blue-50 p-3 rounded-lg text-center">
                    <div className="font-bold text-blue-900">{results.totalFeatures}</div>
                    <div className="text-xs text-blue-800">Total Features</div>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg text-center">
                    <div className="font-bold text-green-900">{results.syntheticFeatures}</div>
                    <div className="text-xs text-green-800">Synthetic</div>
                  </div>
                  <div className="bg-purple-50 p-3 rounded-lg text-center">
                    <div className="font-bold text-purple-900">{results.primitives.length}</div>
                    <div className="text-xs text-purple-800">Primitives</div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Generated Features (Sample)</h4>
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {results.features.map((feature, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm bg-gray-50 p-2 rounded">
                        <CheckSquare className="h-3 w-3 text-green-500" />
                        <span className="font-mono text-xs">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Entity Relationships</h4>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    {results.relationships.map((rel, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm mb-1">
                        <span className="font-medium">{rel.from}</span>
                        <span className="text-gray-500">→</span>
                        <span className="font-medium">{rel.to}</span>
                        <Badge variant="outline" className="text-xs">
                          {rel.type}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                  <h4 className="font-semibold text-green-900 mb-1">✅ Feature Generation Complete</h4>
                  <p className="text-green-800 text-sm">
                    Successfully generated {results.syntheticFeatures} new features using {results.primitives.length}{" "}
                    primitives across {entityCount} entities with depth {maxDepth[0]}.
                  </p>
                </div>
              </div>
            ) : (
              <div className="h-64 bg-gray-100 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <Brain className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-500">Configure settings and generate features</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function PandasPlayground() {
  const [pipeline, setPipeline] = useState([])
  const [code, setCode] = useState("")
  const [isExecuting, setIsExecuting] = useState(false)
  const [output, setOutput] = useState("")

  const sampleDataset = {
    name: "sales_data.csv",
    rows: 5000,
    columns: ["date", "product", "category", "price", "quantity", "customer_age", "region"],
    issues: ["missing values in customer_age", "inconsistent date formats", "outliers in price"],
  }

  const availableOperations = [
    { id: "load", name: "Load Data", icon: Database, code: 'df = pd.read_csv("sales_data.csv")' },
    { id: "info", name: "Data Info", icon: Search, code: "print(df.info())\nprint(df.describe())" },
    {
      id: "missing",
      name: "Handle Missing",
      icon: AlertTriangle,
      code: 'df["customer_age"].fillna(df["customer_age"].median(), inplace=True)',
    },
    {
      id: "scale",
      name: "Scale Features",
      icon: TrendingUp,
      code: 'from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndf[["price", "quantity"]] = scaler.fit_transform(df[["price", "quantity"]])',
    },
    {
      id: "encode",
      name: "Encode Categories",
      icon: Code,
      code: 'df = pd.get_dummies(df, columns=["category", "region"])',
    },
    { id: "filter", name: "Filter Data", icon: Search, code: 'df = df[df["price"] > 0]  # Remove invalid prices' },
    {
      id: "group",
      name: "Group By",
      icon: BarChart3,
      code: 'grouped = df.groupby("category")["price"].agg(["mean", "sum", "count"])',
    },
    {
      id: "datetime",
      name: "Parse Dates",
      icon: Target,
      code: 'df["date"] = pd.to_datetime(df["date"])\ndf["month"] = df["date"].dt.month',
    },
  ]

  const addOperation = (operation) => {
    const newPipeline = [...pipeline, { ...operation, id: Date.now() }]
    setPipeline(newPipeline)
    updateCode(newPipeline)
  }

  const removeOperation = (id) => {
    const newPipeline = pipeline.filter((op) => op.id !== id)
    setPipeline(newPipeline)
    updateCode(newPipeline)
  }

  const updateCode = (currentPipeline) => {
    const imports = `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample dataset: ${sampleDataset.name}
# Shape: ${sampleDataset.rows} rows × ${sampleDataset.columns.length} columns
# Issues: ${sampleDataset.issues.join(", ")}

`

    const pipelineCode = currentPipeline.map((op) => `# ${op.name}\n${op.code}`).join("\n\n")

    const finalCode = `${imports}${pipelineCode}

# Final result
print("Pipeline completed!")
print(f"Final dataset shape: {df.shape}")
print("\\nFirst 5 rows:")
print(df.head())`

    setCode(finalCode)
  }

  const executePipeline = () => {
    setIsExecuting(true)

    setTimeout(() => {
      let simulatedOutput = `Pipeline executed successfully!\n\n`

      pipeline.forEach((op, index) => {
        switch (op.name) {
          case "Load Data":
            simulatedOutput += `Step ${index + 1}: Data loaded - Shape: (5000, 7)\n`
            break
          case "Data Info":
            simulatedOutput += `Step ${index + 1}: Data info displayed\n<class 'pandas.core.frame.DataFrame'>\nRangeIndex: 5000 entries\n`
            break
          case "Handle Missing":
            simulatedOutput += `Step ${index + 1}: Missing values filled - customer_age: 0 missing\n`
            break
          case "Scale Features":
            simulatedOutput += `Step ${index + 1}: Features scaled - price and quantity normalized\n`
            break
          case "Encode Categories":
            simulatedOutput += `Step ${index + 1}: Categories encoded - Shape: (5000, 15)\n`
            break
          case "Filter Data":
            simulatedOutput += `Step ${index + 1}: Data filtered - Removed 23 invalid records\n`
            break
          case "Group By":
            simulatedOutput += `Step ${index + 1}: Groupby completed - 5 categories analyzed\n`
            break
          case "Parse Dates":
            simulatedOutput += `Step ${index + 1}: Dates parsed - Added month column\n`
            break
        }
      })

      simulatedOutput += `\nFinal dataset shape: (${pipeline.some((op) => op.name === "Filter Data") ? "4977" : "5000"}, ${pipeline.some((op) => op.name === "Encode Categories") ? "15" : "7"})\n`
      simulatedOutput += `Pipeline completed successfully!`

      setOutput(simulatedOutput)
      setIsExecuting(false)
    }, 2000)
  }

  const resetPipeline = () => {
    setPipeline([])
    setCode("")
    setOutput("")
  }

  const exportCode = () => {
    const blob = new Blob([code], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "pandas_pipeline.py"
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadNotebook = () => {
    const notebookContent = {
      cells: [
        {
          cell_type: "markdown",
          metadata: {},
          source: [
            "# Pandas Data Processing Pipeline\n",
            "\n",
            "This notebook contains the data processing pipeline created in the Pandas Playground.\n",
            "\n",
            `**Dataset:** ${sampleDataset.name}\n`,
            `**Shape:** ${sampleDataset.rows} rows × ${sampleDataset.columns.length} columns\n`,
            `**Issues:** ${sampleDataset.issues.join(", ")}\n`,
          ],
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [code],
        },
      ],
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3",
        },
        language_info: {
          name: "python",
          version: "3.8.0",
        },
      },
      nbformat: 4,
      nbformat_minor: 4,
    }

    const blob = new Blob([JSON.stringify(notebookContent, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "pandas_pipeline.ipynb"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎛️ Pandas Playground</h3>
        <p className="text-gray-700 mb-4">
          Build data preprocessing pipelines visually using sample sales data and export as Python code.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Pandas</Badge>
          <Badge variant="outline">Pipeline</Badge>
          <Badge variant="outline">Code Export</Badge>
          <Badge variant="outline">Sales Data</Badge>
        </div>
        <div className="mt-4">
          <Button onClick={downloadNotebook} variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Download Jupyter Notebook
          </Button>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Pipeline Builder</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Dataset Info */}
            <div className="bg-blue-50 p-3 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-1">Sample Dataset</h4>
              <p className="text-sm text-blue-800">
                <strong>{sampleDataset.name}</strong> - {sampleDataset.rows.toLocaleString()} rows
              </p>
              <p className="text-xs text-blue-700 mt-1">Columns: {sampleDataset.columns.join(", ")}</p>
            </div>

            {/* Available Operations */}
            <div>
              <h4 className="font-medium mb-2">Available Operations</h4>
              <div className="grid grid-cols-2 gap-2">
                {availableOperations.map((op) => {
                  const Icon = op.icon
                  return (
                    <Button
                      key={op.id}
                      variant="outline"
                      size="sm"
                      className="justify-start h-auto p-2 bg-transparent"
                      onClick={() => addOperation(op)}
                    >
                      <Icon className="h-3 w-3 mr-2" />
                      <span className="text-xs">{op.name}</span>
                    </Button>
                  )
                })}
              </div>
            </div>

            {/* Current Pipeline */}
            <div>
              <h4 className="font-medium mb-2">Current Pipeline ({pipeline.length} steps)</h4>
              <div className="max-h-40 overflow-y-auto space-y-1">
                {pipeline.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-4">Add operations to build your pipeline</p>
                ) : (
                  pipeline.map((op, index) => {
                    const Icon = op.icon
                    return (
                      <div key={op.id} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono bg-blue-100 px-1 rounded">{index + 1}</span>
                          <Icon className="h-3 w-3" />
                          <span className="text-sm">{op.name}</span>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeOperation(op.id)}
                          className="h-6 w-6 p-0"
                        >
                          ×
                        </Button>
                      </div>
                    )
                  })
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button onClick={executePipeline} className="flex-1" disabled={pipeline.length === 0 || isExecuting}>
                {isExecuting ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Execute Pipeline
                  </>
                )}
              </Button>
              <Button onClick={resetPipeline} variant="outline">
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generated Code & Output</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="code" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="code">Generated Code</TabsTrigger>
                <TabsTrigger value="output">Execution Output</TabsTrigger>
              </TabsList>

              <TabsContent value="code" className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Python Code</h4>
                  <Button onClick={exportCode} variant="outline" size="sm" disabled={!code}>
                    <Download className="h-4 w-4 mr-2" />
                    Export .py
                  </Button>
                </div>
                <textarea
                  value={code || "# Add operations to generate code..."}
                  readOnly
                  className="w-full h-64 p-3 font-mono text-sm border rounded-lg bg-gray-900 text-green-400 resize-none"
                />
              </TabsContent>

              <TabsContent value="output" className="space-y-4">
                <h4 className="font-medium">Execution Results</h4>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm min-h-64 whitespace-pre-wrap overflow-auto">
                  {output || "Execute pipeline to see results..."}
                </div>
              </TabsContent>
            </Tabs>

            {pipeline.length > 0 && (
              <div className="mt-4 bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                <h4 className="font-semibold text-green-900 mb-2">Pipeline Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-green-800">
                      <strong>Steps:</strong> {pipeline.length}
                    </p>
                    <p className="text-green-800">
                      <strong>Operations:</strong> {pipeline.map((op) => op.name).join(", ")}
                    </p>
                  </div>
                  <div>
                    <p className="text-green-800">
                      <strong>Code Lines:</strong> {code.split("\n").length}
                    </p>
                    <p className="text-green-800">
                      <strong>Status:</strong> {output ? "Executed" : "Ready to run"}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

const courseModules = [
  {
    id: "module-2",
    title: "Module 2: Data Engineering",
    description: "Advanced data preprocessing, feature engineering, and pipeline development",
    status: "active" as const,
    progress: 0,
    sections: [
      { id: "data-engineering-intro", title: "Understanding Data Engineering in ML", icon: Database, completed: false },
      { id: "missing-data", title: "Handling Missing Data", icon: Search, completed: false },
      { id: "imbalanced-datasets", title: "Working with Imbalanced Datasets", icon: BarChart3, completed: false },
      { id: "feature-stores", title: "Introduction to Feature Stores", icon: Cloud, completed: false },
      { id: "featuretools", title: "Feature Engineering with Featuretools", icon: Zap, completed: false },
      { id: "toolset-overview", title: "Toolset: Pandas, Imblearn, Featuretools", icon: Code, completed: false },
      { id: "missing-data-dashboard", title: "🎛️ Missingness Map Dashboard", icon: Target, completed: false },
      { id: "imbalance-dashboard", title: "🎛️ Imbalance Fixer Studio", icon: BarChart3, completed: false },
      { id: "feature-synthesis-dashboard", title: "🎛️ Feature Synthesizer", icon: Brain, completed: false },
      { id: "pandas-pipeline-dashboard", title: "🎛️ Pandas Playground", icon: FileCode, completed: false },
      { id: "fraud-detection-project", title: "💻 Credit Card Fraud Detection", icon: Code, completed: false },
    ],
  },
]

export default function ELearningPlatform() {
  const [activeModule, setActiveModule] = useState("module-2")
  const [activeSection, setActiveSection] = useState("data-engineering-intro")
  const [completedSections, setCompletedSections] = useState<Record<string, string[]>>({
    "module-2": [],
  })
  const [showAssignment, setShowAssignment] = useState(false)

  const currentModule = courseModules[0]
  const currentSections = currentModule.sections

  const markComplete = (sectionId: string) => {
    if (!completedSections[activeModule].includes(sectionId)) {
      setCompletedSections((prev) => ({
        ...prev,
        [activeModule]: [...prev[activeModule], sectionId],
      }))
    }
  }

  const getModuleProgress = (moduleId: string) => {
    const completed = completedSections[moduleId].length
    const total = currentSections.length
    return (completed / total) * 100
  }

  const currentProgress = getModuleProgress(activeModule)

  return (
    <SidebarProvider defaultOpen={true}>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex">
        <Sidebar collapsible="icon" className="border-r">
          <SidebarHeader className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <GraduationCap className="h-6 w-6 text-blue-600 flex-shrink-0" />
              <h2 className="text-lg font-bold text-sidebar-foreground group-data-[collapsible=icon]:hidden">
                {currentModule.title}
              </h2>
            </div>
            <div className="group-data-[collapsible=icon]:hidden">
              <p className="text-sm text-sidebar-foreground/70">
                {completedSections[activeModule].length} of {currentSections.length} sections completed
              </p>
              <Progress value={currentProgress} className="h-2 mt-2" />
              <div className="flex justify-between text-xs text-sidebar-foreground/70 mt-1">
                <span>Progress</span>
                <span>{Math.round(currentProgress)}%</span>
              </div>
            </div>
          </SidebarHeader>
          <SidebarContent className="p-0">
            <ScrollArea className="h-[calc(100vh-200px)]">
              <SidebarMenu className="p-2">
                {currentSections.map((section, index) => {
                  const Icon = section.icon
                  const isCompleted = completedSections[activeModule].includes(section.id)
                  const isActive = activeSection === section.id

                  return (
                    <SidebarMenuItem key={section.id}>
                      <SidebarMenuButton
                        isActive={isActive}
                        className="w-full justify-start text-left h-auto p-3"
                        onClick={() => {
                          setActiveSection(section.id)
                          setShowAssignment(false)
                        }}
                        tooltip={section.title}
                      >
                        <div className="flex items-start gap-3 w-full">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded group-data-[collapsible=icon]:hidden">
                              {String(index + 1).padStart(2, "0")}
                            </span>
                            {isCompleted ? (
                              <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                            ) : (
                              <Circle className="h-4 w-4 flex-shrink-0" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0 group-data-[collapsible=icon]:hidden">
                            <div className="flex items-center gap-2">
                              <Icon className="h-4 w-4 flex-shrink-0" />
                              <span className="font-medium text-sm truncate">{section.title}</span>
                            </div>
                          </div>
                        </div>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )
                })}
              </SidebarMenu>
            </ScrollArea>
          </SidebarContent>
          <SidebarFooter className="pt-4 border-t border-sidebar-border p-2">
            <Button
              variant={showAssignment ? "default" : "outline"}
              className="w-full justify-start"
              onClick={() => setShowAssignment(true)}
            >
              <Award className="h-4 w-4 mr-2 flex-shrink-0" />
              <span className="group-data-[collapsible=icon]:hidden">Module Assessment</span>
            </Button>
          </SidebarFooter>
        </Sidebar>

        <SidebarInset className="flex-1">
          <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4 bg-white">
            <SidebarTrigger className="-ml-1" />
            <div className="flex items-center gap-2">
              <GraduationCap className="h-6 w-6 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900">Module 2: Data Engineering</h1>
            </div>
          </header>
          <div className="flex-1 p-6 overflow-auto">
            {!showAssignment ? (
              <ContentSection
                activeSection={activeSection}
                activeModule={activeModule}
                onComplete={markComplete}
                isCompleted={completedSections[activeModule].includes(activeSection)}
              />
            ) : (
              <AssignmentSection moduleTitle={currentModule.title} onBack={() => setShowAssignment(false)} />
            )}
          </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}

function ContentSection({
  activeSection,
  activeModule,
  onComplete,
  isCompleted,
}: {
  activeSection: string
  activeModule: string
  onComplete: (id: string) => void
  isCompleted: boolean
}) {
  const content = getContentForSection(activeSection)

  return (
    <Card className="min-h-[600px]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline" className="text-xs">
                MODULE 2
              </Badge>
              <span className="text-sm text-gray-600">2. {content.title}</span>
            </div>
            <CardTitle className="text-2xl">{content.title}</CardTitle>
            <CardDescription className="text-lg mt-2">{content.description}</CardDescription>
          </div>
          {!isCompleted && (
            <Button onClick={() => onComplete(activeSection)} className="ml-4">
              <CheckCircle className="h-4 w-4 mr-2" />
              Mark Complete
            </Button>
          )}
          {isCompleted && (
            <Badge variant="secondary" className="ml-4">
              <CheckCircle className="h-4 w-4 mr-2" />
              Completed
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="prose max-w-none">{content.content}</div>
      </CardContent>
    </Card>
  )
}

function AssignmentSection({ moduleTitle, onBack }: { moduleTitle: string; onBack: () => void }) {
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [showResults, setShowResults] = useState(false)

  const mcqQuestions = [
    {
      id: "q1",
      question: "What is the purpose of a feature store?",
      options: ["Store models", "Store engineered features", "Store datasets", "Store predictions"],
      correct: "Store engineered features",
    },
    {
      id: "q2",
      question: "Which library is used for oversampling techniques like SMOTE?",
      options: ["Pandas", "Scikit-learn", "Imblearn", "Featuretools"],
      correct: "Imblearn",
    },
    {
      id: "q3",
      question: "What type of missingness is hardest to deal with?",
      options: ["MCAR", "MAR", "MNAR", "N/A"],
      correct: "MNAR",
    },
    {
      id: "q4",
      question: "What does Featuretools use to generate features?",
      options: ["Label encoding", "Deep Feature Synthesis", "PCA", "SQL Joins"],
      correct: "Deep Feature Synthesis",
    },
    {
      id: "q5",
      question: "Which metric is best to evaluate imbalanced classification models?",
      options: ["Accuracy", "Precision", "F1-score", "R²"],
      correct: "F1-score",
    },
    {
      id: "q6",
      question: "What does SMOTE stand for?",
      options: [
        "Simple Minority Oversampling Technique",
        "Synthetic Minority Oversampling Technique",
        "Statistical Minority Oversampling Technique",
        "Systematic Minority Oversampling Technique",
      ],
      correct: "Synthetic Minority Oversampling Technique",
    },
    {
      id: "q7",
      question: "Which imputation strategy is best for categorical data?",
      options: ["Mean", "Median", "Mode", "Forward fill"],
      correct: "Mode",
    },
    {
      id: "q8",
      question: "What is the main advantage of using a feature store?",
      options: ["Faster training", "Feature reusability", "Better accuracy", "Reduced storage"],
      correct: "Feature reusability",
    },
    {
      id: "q9",
      question: "Which technique creates synthetic samples by interpolation?",
      options: ["Random oversampling", "SMOTE", "Undersampling", "Tomek links"],
      correct: "SMOTE",
    },
    {
      id: "q10",
      question: "What is the primary goal of data engineering in ML?",
      options: ["Model training", "Data visualization", "Preparing model-ready data", "Feature selection"],
      correct: "Preparing model-ready data",
    },
  ]

  const fillInBlanks = [
    {
      id: "f1",
      question: "_______ is a technique to generate synthetic minority samples.",
      answer: "SMOTE",
    },
    {
      id: "f2",
      question: "The Python library _______ is used for feature synthesis.",
      answer: "Featuretools",
    },
    {
      id: "f3",
      question: "A _______ is a system for storing and sharing ML features.",
      answer: "feature store",
    },
    {
      id: "f4",
      question: "Missing data can be filled using _______ or regression-based techniques.",
      answer: "imputation",
    },
    {
      id: "f5",
      question: "_______ datasets can lead to biased model performance.",
      answer: "Imbalanced",
    },
    {
      id: "f6",
      question: "_______ is used for data manipulation and analysis in Python.",
      answer: "Pandas",
    },
    {
      id: "f7",
      question: "The _______ library provides tools for handling imbalanced datasets.",
      answer: "Imblearn",
    },
    {
      id: "f8",
      question: "_______ Feature Synthesis is used by Featuretools to generate features.",
      answer: "Deep",
    },
    {
      id: "f9",
      question: "Data engineering ensures data is _______ for machine learning models.",
      answer: "model-ready",
    },
    {
      id: "f10",
      question: "_______ imputation uses the most frequent value to fill missing data.",
      answer: "Mode",
    },
  ]

  const handleSubmit = () => {
    setShowResults(true)
  }

  const getScore = () => {
    let correct = 0
    mcqQuestions.forEach((q) => {
      if (answers[q.id] === q.correct) correct++
    })
    fillInBlanks.forEach((q) => {
      if (answers[q.id]?.toLowerCase().includes(q.answer.toLowerCase())) correct++
    })
    return correct
  }

  return (
    <Card className="min-h-[600px]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline" className="text-xs">
                MODULE ASSESSMENT
              </Badge>
            </div>
            <CardTitle className="text-2xl">{moduleTitle} - Final Assessment</CardTitle>
            <CardDescription className="text-lg mt-2">Test your understanding with 20 questions</CardDescription>
          </div>
          <Button variant="outline" onClick={onBack}>
            Back to Content
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="mcq" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="mcq">Multiple Choice (10 Questions)</TabsTrigger>
            <TabsTrigger value="fill">Fill in the Blanks (10 Questions)</TabsTrigger>
          </TabsList>

          <TabsContent value="mcq" className="space-y-6">
            {mcqQuestions.map((question, index) => (
              <Card key={question.id}>
                <CardHeader>
                  <CardTitle className="text-lg">Question {index + 1}</CardTitle>
                  <CardDescription>{question.question}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {question.options.map((option) => (
                      <Button
                        key={option}
                        variant={answers[question.id] === option ? "default" : "outline"}
                        className="w-full justify-start"
                        onClick={() => setAnswers({ ...answers, [question.id]: option })}
                      >
                        {option}
                      </Button>
                    ))}
                  </div>
                  {showResults && (
                    <div className="mt-4 p-3 rounded-lg bg-gray-50">
                      <p
                        className={`font-medium ${answers[question.id] === question.correct ? "text-green-600" : "text-red-600"}`}
                      >
                        {answers[question.id] === question.correct ? "✓ Correct!" : "✗ Incorrect"}
                      </p>
                      {answers[question.id] !== question.correct && (
                        <p className="text-sm text-gray-600 mt-1">Correct answer: {question.correct}</p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          <TabsContent value="fill" className="space-y-6">
            {fillInBlanks.map((question, index) => (
              <Card key={question.id}>
                <CardHeader>
                  <CardTitle className="text-lg">Question {index + 11}</CardTitle>
                  <CardDescription>{question.question}</CardDescription>
                </CardHeader>
                <CardContent>
                  <input
                    type="text"
                    className="w-full p-2 border rounded-md"
                    placeholder="Enter your answer..."
                    value={answers[question.id] || ""}
                    onChange={(e) => setAnswers({ ...answers, [question.id]: e.target.value })}
                  />
                  {showResults && (
                    <div className="mt-4 p-3 rounded-lg bg-gray-50">
                      <p
                        className={`font-medium ${answers[question.id]?.toLowerCase().includes(question.answer.toLowerCase()) ? "text-green-600" : "text-red-600"}`}
                      >
                        {answers[question.id]?.toLowerCase().includes(question.answer.toLowerCase())
                          ? "✓ Correct!"
                          : "✗ Incorrect"}
                      </p>
                      <p className="text-sm text-gray-600 mt-1">Expected answer: {question.answer}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          <div className="mt-8 flex items-center justify-between">
            {!showResults ? (
              <Button onClick={handleSubmit} size="lg">
                Submit Assessment
              </Button>
            ) : (
              <div className="flex items-center gap-4">
                <Badge variant="secondary" className="text-lg p-2">
                  Score: {getScore()}/20 ({Math.round((getScore() / 20) * 100)}%)
                </Badge>
                <Button
                  onClick={() => {
                    setShowResults(false)
                    setAnswers({})
                  }}
                  variant="outline"
                >
                  Retake Assessment
                </Button>
              </div>
            )}
          </div>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function getContentForSection(sectionId: string) {
  const contentMap: Record<string, any> = {
    "data-engineering-intro": {
      title: "Understanding Data Engineering in ML",
      description: "The foundation of preparing raw data for machine learning models",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Overview</h3>
            <p className="text-gray-700 leading-relaxed">
              Data engineering in machine learning is the process of preparing raw data into a usable format for model
              training. This includes data transformation, dealing with missing or imbalanced datasets, and storing
              engineered features efficiently.
            </p>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-900 mb-2">Key Focus Areas</h4>
            <ul className="text-blue-800 space-y-1">
              <li>• Enhancing data quality, consistency, and structure</li>
              <li>• Ensuring the data is model-ready</li>
              <li>• Building scalable ML workflow foundations</li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">The Data Engineering Pipeline</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Data Collection</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Gathering raw data from various sources</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Data Processing</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Cleaning, transforming, and preparing data</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Feature Engineering</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Creating meaningful features for ML models</p>
                </CardContent>
              </Card>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Real-World Example: E-commerce Recommendation System</h3>
            <div className="bg-green-50 p-4 rounded-lg">
              <p className="text-green-800 mb-3">
                <strong>Scenario:</strong> Building a product recommendation system for an online store
              </p>
              <div className="space-y-2 text-green-700 text-sm">
                <p>
                  <strong>Raw Data Sources:</strong>
                </p>
                <ul className="list-disc list-inside ml-4 space-y-1">
                  <li>User clickstream data (messy, real-time)</li>
                  <li>Product catalog (inconsistent categories)</li>
                  <li>Purchase history (missing user demographics)</li>
                  <li>Reviews and ratings (unstructured text)</li>
                </ul>
                <p className="mt-3">
                  <strong>Data Engineering Tasks:</strong>
                </p>
                <ul className="list-disc list-inside ml-4 space-y-1">
                  <li>Clean and standardize product categories</li>
                  <li>Handle missing user information</li>
                  <li>Create features like "average rating", "purchase frequency"</li>
                  <li>Process text reviews for sentiment analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      ),
    },

    "missing-data": {
      title: "Handling Missing Data",
      description: "Strategies and techniques for dealing with incomplete datasets",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Understanding Missing Data</h3>
            <p className="text-gray-700 leading-relaxed">
              Missing data is a common challenge in real-world datasets. The approach to handling missing data depends
              on the pattern and severity of the missingness. Understanding why data is missing is crucial for choosing
              the right strategy.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Types of Missingness with Examples</h3>
            <div className="grid md:grid-cols-1 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-green-700">MCAR - Missing Completely at Random</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-2">
                    Missing values are completely random and unrelated to any other data
                  </p>
                  <div className="bg-green-50 p-3 rounded-lg">
                    <p className="text-green-800 text-sm">
                      <strong>Example:</strong> A survey where some respondents accidentally skip questions due to a
                      website glitch. The missingness has nothing to do with the respondent's characteristics or other
                      answers.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-yellow-700">MAR - Missing at Random</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-2">
                    Missing values depend on observed data but not on the missing values themselves
                  </p>
                  <div className="bg-yellow-50 p-3 rounded-lg">
                    <p className="text-yellow-800 text-sm">
                      <strong>Example:</strong> In a health study, younger people are less likely to report their
                      income. The missingness in income depends on age (which we observe), but not on the actual income
                      value.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-red-700">MNAR - Missing Not at Random</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-2">
                    Missing values depend on the unobserved values themselves
                  </p>
                  <div className="bg-red-50 p-3 rounded-lg">
                    <p className="text-red-800 text-sm">
                      <strong>Example:</strong> People with very high incomes refuse to disclose their salary in a
                      survey. The missingness depends on the actual income value (which we don't observe). This is the
                      hardest to handle!
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Common Strategies Explained</h3>
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">1. Deletion Methods</h4>
                <ul className="text-blue-800 text-sm space-y-1">
                  <li>
                    <strong>Listwise deletion:</strong> Remove entire rows with any missing values
                  </li>
                  <li>
                    <strong>Pairwise deletion:</strong> Use all available data for each analysis
                  </li>
                  <li>
                    <strong>When to use:</strong> When data is MCAR and you have plenty of data
                  </li>
                </ul>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">2. Imputation Methods</h4>
                <ul className="text-green-800 text-sm space-y-1">
                  <li>
                    <strong>Mean/Median:</strong> Replace with average (numerical data)
                  </li>
                  <li>
                    <strong>Mode:</strong> Replace with most frequent value (categorical data)
                  </li>
                  <li>
                    <strong>Forward/Backward fill:</strong> Use previous/next value (time series)
                  </li>
                  <li>
                    <strong>KNN Imputation:</strong> Use similar records to estimate missing values
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Practical Example: Customer Dataset</h3>

            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm space-y-2">
              <div>import pandas as pd</div>
              <div>from sklearn.impute import SimpleImputer, KNNImputer</div>
              <div></div>
              <div># Sample customer data with missing values</div>
              <div>{"data = {'age': [25, 30, null, 45, null],"}</div>
              <div>{" 'income': [50000, null, 75000, null, 90000],"}</div>
              <div>{" 'education': ['Bachelor', 'Master', null, 'PhD', 'Bachelor']}"}</div>
              <div>df = pd.DataFrame(data)</div>
              <div></div>
              <div># Strategy 1: Mean imputation for numerical data</div>
              <div>{"age_imputer = SimpleImputer(strategy='mean')"}</div>
              <div>{"df['age_imputed'] = age_imputer.fit_transform(df[['age']])"}</div>
              <div></div>
              <div># Strategy 2: Mode imputation for categorical data</div>
              <div>{"edu_imputer = SimpleImputer(strategy='most_frequent')"}</div>
              <div>{"df['education_imputed'] = edu_imputer.fit_transform(df[['education']])"}</div>
              <div></div>
              <div># Strategy 3: KNN imputation (considers relationships)</div>
              <div>knn_imputer = KNNImputer(n_neighbors=2)</div>
              <div>{"df_knn = pd.DataFrame(knn_imputer.fit_transform(df[['age', 'income']]),"}</div>
              <div>{" columns=['age_knn', 'income_knn'])"}</div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Choosing the Right Strategy</h3>
            <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
              <h4 className="font-semibold text-yellow-900 mb-2">Decision Framework:</h4>
              <ol className="text-yellow-800 text-sm space-y-2 list-decimal list-inside">
                <li>
                  <strong>Analyze the pattern:</strong> Is it MCAR, MAR, or MNAR?
                </li>
                <li>
                  <strong>Check the percentage:</strong> Less than 5% missing → simple imputation, more than 20% →
                  consider deletion
                </li>
                <li>
                  <strong>Consider the feature importance:</strong> Critical features need careful handling
                </li>
                <li>
                  <strong>Test multiple strategies:</strong> Compare model performance with different approaches
                </li>
              </ol>
            </div>
          </div>
        </div>
      ),
    },

    "imbalanced-datasets": {
      title: "Working with Imbalanced Datasets",
      description: "Techniques for handling datasets with unequal class distributions",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Understanding Class Imbalance</h3>
            <p className="text-gray-700 leading-relaxed">
              Class imbalance occurs when one class significantly outnumbers another in your dataset. This is extremely
              common in real-world scenarios and can severely impact model performance if not handled properly.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Real-World Examples of Imbalanced Data</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-red-700">Fraud Detection</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <p>
                      <strong>Scenario:</strong> Credit card transactions
                    </p>
                    <p>
                      <strong>Imbalance:</strong> 99.8% legitimate, 0.2% fraud
                    </p>
                    <p>
                      <strong>Problem:</strong> Model predicts "no fraud" for everything and gets 99.8% accuracy!
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-blue-700">Medical Diagnosis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <p>
                      <strong>Scenario:</strong> Cancer screening
                    </p>
                    <p>
                      <strong>Imbalance:</strong> 95% healthy, 5% positive cases
                    </p>
                    <p>
                      <strong>Problem:</strong> Missing positive cases can be life-threatening!
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-green-700">Email Classification</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <p>
                      <strong>Scenario:</strong> Spam detection
                    </p>
                    <p>
                      <strong>Imbalance:</strong> 85% legitimate, 15% spam
                    </p>
                    <p>
                      <strong>Problem:</strong> Important emails might be missed
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Common Techniques to Address Imbalance</h3>
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">1. Resampling Techniques</h4>
                <ul className="text-blue-800 text-sm space-y-1">
                  <li>
                    <strong>Oversampling:</strong> Duplicate minority class samples
                  </li>
                  <li>
                    <strong>Undersampling:</strong> Remove majority class samples
                  </li>
                  <li>
                    <strong>SMOTE:</strong> Create synthetic minority class samples
                  </li>
                </ul>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">2. Cost-Sensitive Learning</h4>
                <ul className="text-green-800 text-sm space-y-1">
                  <li>
                    <strong>Assign higher misclassification costs:</strong> Penalize errors on the minority class more
                  </li>
                  <li>
                    <strong>Adjust decision thresholds:</strong> Optimize for recall instead of overall accuracy
                  </li>
                </ul>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-semibold text-yellow-900 mb-2">3. Ensemble Methods</h4>
                <ul className="text-yellow-800 text-sm space-y-1">
                  <li>
                    <strong>Bagging/Boosting:</strong> Combine multiple models to improve robustness
                  </li>
                  <li>
                    <strong>Balanced Random Forest:</strong> Use undersampling within each tree
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Practical Example: Fraud Detection</h3>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm space-y-2">
              <div>import pandas as pd</div>
              <div>from imblearn.over_sampling import SMOTE</div>
              <div>from sklearn.model_selection import train_test_split</div>
              <div>from sklearn.linear_model import LogisticRegression</div>
              <div>from sklearn.metrics import classification_report</div>
              <div></div>
              <div># Load fraud dataset (highly imbalanced)</div>
              <div>df = pd.read_csv('fraud_data.csv')</div>
              <div>X = df.drop('Class', axis=1)</div>
              <div>y = df['Class']</div>
              <div></div>
              <div># Split into train and test sets</div>
              <div>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)</div>
              <div></div>
              <div># Apply SMOTE to oversample minority class</div>
              <div>smote = SMOTE(random_state=42)</div>
              <div>X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)</div>
              <div></div>
              <div># Train Logistic Regression model</div>
              <div>model = LogisticRegression()</div>
              <div>model.fit(X_train_smote, y_train_smote)</div>
              <div></div>
              <div># Evaluate performance</div>
              <div>y_pred = model.predict(X_test)</div>
              <div>print(classification_report(y_test, y_pred))</div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Choosing the Right Technique</h3>
            <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
              <h4 className="font-semibold text-yellow-900 mb-2">Decision Framework:</h4>
              <ol className="text-yellow-800 text-sm space-y-2 list-decimal list-inside">
                <li>
                  <strong>Assess the imbalance ratio:</strong> How severe is the imbalance?
                </li>
                <li>
                  <strong>Consider the dataset size:</strong> Small datasets benefit from SMOTE
                </li>
                <li>
                  <strong>Evaluate the model performance:</strong> Use precision, recall, F1-score
                </li>
                <li>
                  <strong>Experiment with multiple techniques:</strong> Find the best approach for your data
                </li>
              </ol>
            </div>
          </div>
        </div>
      ),
    },

    "feature-stores": {
      title: "Introduction to Feature Stores",
      description: "Centralized repositories for storing and managing machine learning features",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">What is a Feature Store?</h3>
            <p className="text-gray-700 leading-relaxed">
              A feature store is a centralized repository for storing, managing, and serving machine learning features.
              It acts as a single source of truth for features, ensuring consistency and reusability across different ML
              models and teams.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Key Benefits of Using a Feature Store</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-blue-700">Feature Reusability</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Features can be reused across multiple models, reducing redundancy and development time.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-green-700">Consistency</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Ensures consistent feature definitions and transformations across training and serving environments.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-yellow-700">Scalability</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Handles large volumes of feature data and provides low-latency access for real-time predictions.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-red-700">Governance</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Provides a centralized platform for managing feature access, lineage, and monitoring.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Components of a Feature Store</h3>
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">1. Feature Definition</h4>
                <p className="text-blue-800 text-sm">Defines the schema, data type, and metadata for each feature.</p>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">2. Feature Ingestion</h4>
                <p className="text-green-800 text-sm">
                  Ingests feature data from various sources, such as databases, data warehouses, and streaming
                  platforms.
                </p>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-semibold text-yellow-900 mb-2">3. Feature Storage</h4>
                <p className="text-yellow-800 text-sm">
                  Stores feature data in both online and offline storage systems.
                </p>
              </div>

              <div className="bg-red-50 p-4 rounded-lg">
                <h4 className="font-semibold text-red-900 mb-2">4. Feature Serving</h4>
                <p className="text-red-800 text-sm">
                  Provides low-latency access to features for real-time predictions.
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Practical Example: Customer Churn Prediction</h3>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm space-y-2">
              <div># Define customer features</div>
              <div>features = [</div>
              <div> {"{'name': 'customer_age', 'type': 'int'},"}</div>
              <div> {"{'name': 'total_purchases', 'type': 'int'},"}</div>
              <div> {"{'name': 'avg_transaction_value', 'type': 'float'},"}</div>
              <div> {"{'name': 'days_since_last_purchase', 'type': 'int'}"}</div>
              <div>]</div>
              <div></div>
              <div># Ingest features from database</div>
              <div>def ingest_features(customer_id, features):</div>
              <div> # Connect to database</div>
              <div> # Extract feature values for customer_id</div>
              <div> # Store features in feature store</div>
              <div> pass</div>
              <div></div>
              <div># Serve features for model prediction</div>
              <div>def get_features(customer_id):</div>
              <div> # Retrieve features from feature store</div>
              <div> return features</div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Popular Feature Store Platforms</h3>
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">1. Feast</h4>
                <p className="text-blue-800 text-sm">
                  An open-source feature store for managing and serving machine learning features.
                </p>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">2. Tecton</h4>
                <p className="text-green-800 text-sm">
                  A commercial feature store platform for building and deploying real-time ML applications.
                </p>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-semibold text-yellow-900 mb-2">3. Hopsworks</h4>
                <p className="text-yellow-800 text-sm">A data platform for AI with an integrated feature store.</p>
              </div>
            </div>
          </div>
        </div>
      ),
    },

    featuretools: {
      title: "Feature Engineering with Featuretools",
      description: "Automated feature engineering using the Featuretools library",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">What is Featuretools?</h3>
            <p className="text-gray-700 leading-relaxed">
              Featuretools is a Python library for automated feature engineering. It automatically creates new features
              from relational datasets using a technique called Deep Feature Synthesis (DFS).
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Key Concepts in Featuretools</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-blue-700">Entities</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Tables in a relational dataset, such as customers, transactions, and products.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-green-700">Relationships</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Connections between entities, such as a customer having multiple transactions.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-yellow-700">Primitives</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    Operations that can be applied to entities and relationships to create new features, such as SUM,
                    MEAN, and COUNT.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg text-red-700">Deep Feature Synthesis (DFS)</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">
                    A technique for automatically creating new features by combining primitives across multiple entities
                    and relationships.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Practical Example: E-commerce Dataset</h3>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm space-y-2">
              <div>import featuretools as ft</div>
              <div>import pandas as pd</div>
              <div></div>
              <div># Load e-commerce data</div>
              <div>customers = pd.read_csv('customers.csv')</div>
              <div>transactions = pd.read_csv('transactions.csv')</div>
              <div></div>
              <div># Create entityset</div>
              <div>es = ft.EntitySet(id='ecom')</div>
              <div></div>
              <div># Add entities</div>
              <div>es = es.add_dataframe(dataframe_name='customers', dataframe=customers, index='customer_id')</div>
              <div>
                es = es.add_dataframe(dataframe_name='transactions', dataframe=transactions, index='transaction_id',
                time_index='transaction_time')
              </div>
              <div></div>
              <div># Add relationship</div>
              <div>
                es = es.add_relationship(ft.Relationship(customers['customer_id'], transactions['customer_id']))
              </div>
              <div></div>
              <div># Run DFS</div>
              <div>
                feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='customers', max_depth=2,
                verbose=True)
              </div>
              <div></div>
              <div># Print feature matrix</div>
              <div>print(feature_matrix.head())</div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Benefits of Using Featuretools</h3>
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">1. Automation</h4>
                <p className="text-blue-800 text-sm">
                  Automatically creates new features, reducing the need for manual feature engineering.
                </p>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">2. Efficiency</h4>
                <p className="text-green-800 text-sm">
                  Quickly generates a large number of features, allowing you to explore a wider range of possibilities.
                </p>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-semibold text-yellow-900 mb-2">3. Consistency</h4>
                <p className="text-yellow-800 text-sm">
                  Ensures consistent feature definitions and transformations across different datasets.
                </p>
              </div>
            </div>
          </div>
        </div>
      ),
    },

    "toolset-overview": {
      title: "Toolset: Pandas, Imblearn, Featuretools",
      description: "Overview of the key libraries used for data engineering",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Pandas</h3>
            <p className="text-gray-700 leading-relaxed">
              Pandas is a powerful Python library for data manipulation and analysis. It provides data structures for
              efficiently storing and manipulating large datasets, as well as tools for cleaning, transforming, and
              analyzing data.
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>
                <strong>DataFrames:</strong> Two-dimensional labeled data structures with columns of potentially
                different types.
              </li>
              <li>
                <strong>Series:</strong> One-dimensional labeled array capable of holding any data type.
              </li>
              <li>
                <strong>Data Cleaning:</strong> Tools for handling missing values, duplicates, and outliers.
              </li>
              <li>
                <strong>Data Transformation:</strong> Functions for filtering, sorting, grouping, and aggregating data.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Imblearn</h3>
            <p className="text-gray-700 leading-relaxed">
              Imblearn is a Python library for handling imbalanced datasets. It provides a variety of techniques for
              oversampling the minority class and undersampling the majority class, as well as ensemble methods for
              improving the performance of machine learning models on imbalanced data.
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>
                <strong>Oversampling:</strong> Techniques for creating synthetic samples of the minority class, such as
                SMOTE and ADASYN.
              </li>
              <li>
                <strong>Undersampling:</strong> Methods for removing samples from the majority class, such as
                RandomUnderSampler and TomekLinks.
              </li>
              <li>
                <strong>Ensemble Methods:</strong> Algorithms for combining multiple models to improve performance on
                imbalanced data, such as BalancedRandomForestClassifier and EasyEnsembleClassifier.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Featuretools</h3>
            <p className="text-gray-700 leading-relaxed">
              Featuretools is a Python library for automated feature engineering. It automatically creates new features
              from relational datasets using a technique called Deep Feature Synthesis (DFS).
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>
                <strong>Entities:</strong> Tables in a relational dataset, such as customers, transactions, and
                products.
              </li>
              <li>
                <strong>Relationships:</strong> Connections between entities, such as a customer having multiple
                transactions.
              </li>
              <li>
                <strong>Primitives:</strong> Operations that can be applied to entities and relationships to create new
                features, such as SUM, MEAN, and COUNT.
              </li>
              <li>
                <strong>Deep Feature Synthesis (DFS):</strong> A technique for automatically creating new features by
                combining primitives across multiple entities and relationships.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Practical Example: Combining the Toolset</h3>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm space-y-2">
              <div>import pandas as pd</div>
              <div>from imblearn.over_sampling import SMOTE</div>
              <div>import featuretools as ft</div>
              <div></div>
              <div># Load data using Pandas</div>
              <div>customers = pd.read_csv('customers.csv')</div>
              <div>transactions = pd.read_csv('transactions.csv')</div>
              <div></div>
              <div># Create entityset using Featuretools</div>
              <div>es = ft.EntitySet(id='ecom')</div>
              <div>es = es.add_dataframe(dataframe_name='customers', dataframe=customers, index='customer_id')</div>
              <div>
                es = es.add_dataframe(dataframe_name='transactions', dataframe=transactions, index='transaction_id',
                time_index='transaction_time')
              </div>
              <div></div>
              <div># Add relationship</div>
              <div>
                es = es.add_relationship(ft.Relationship(customers['customer_id'], transactions['customer_id']))
              </div>
              <div></div>
              <div># Run DFS to generate features</div>
              <div>
                feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='customers', max_depth=2,
                verbose=True)
              </div>
              <div></div>
              <div># Handle imbalanced data using Imblearn</div>
              <div>smote = SMOTE(random_state=42)</div>
              <div>X_smote, y_smote = smote.fit_resample(feature_matrix, labels)</div>
            </div>
          </div>
        </div>
      ),
    },

    "missing-data-dashboard": {
      title: "🎛️ Missingness Map Dashboard",
      description: "Interactive tool to visualize and handle missing data patterns",
      content: <MissingDataDashboard />,
    },

    "imbalance-dashboard": {
      title: "🎛️ Imbalance Fixer Studio",
      description: "Interactive tool to balance imbalanced datasets using various resampling techniques",
      content: <ImbalanceDashboard />,
    },

    "feature-synthesis-dashboard": {
      title: "🎛️ Feature Synthesizer",
      description: "Automatically generate features from relational e-commerce data using Deep Feature Synthesis (DFS)",
      content: <FeatureSynthesisDashboard />,
    },

    "pandas-pipeline-dashboard": {
      title: "🎛️ Pandas Playground",
      description: "Build data preprocessing pipelines visually using sample sales data and export as Python code",
      content: <PandasPlayground />,
    },

    "fraud-detection-project": {
      title: "💻 Credit Card Fraud Detection",
      description:
        "End-to-end project demonstrating how to handle severe class imbalance in fraud detection using SMOTE",
      content: <FraudDetectionProject />,
    },
  }

  return (
    contentMap[sectionId] || {
      title: "Content Not Found",
      description: "The content for this section is not available.",
      content: <div>Content not available</div>,
    }
  )
}
