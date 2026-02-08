"use client"

import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle, AlertTriangle, BarChart, Activity, PieChart as PieIcon, Search, ThumbsUp, ThumbsDown } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart as RechartsBarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';

// --- Types ---
interface AnalysisSummary {
    total_reviews: number;
    fake_reviews: number;
    genuine_reviews: number;
    avg_trust_score: number;
}

interface AspectReport {
    avg_weighted_score: number;
    dominant_sentiment: string;
    dominant_emotion: string;
    confidence_score: number;
    evidence: Array<{ text: string, trust_score: number, emotion: string }>;
    summary: string;
}

interface Review {
    id: number;
    Text: string;
    trust_score: number;
    review_label: string;
    fake_reason: string;
    emotion_intensity: number;
    emotional_exaggeration: number;
    // ... other fields
}

interface AnalysisResult {
    summary: AnalysisSummary;
    reviews: Review[];
    aspects: Record<string, AspectReport>;
}

export default function FakeReviewDetector() {
    const [file, setFile] = useState<File | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [results, setResults] = useState<AnalysisResult | null>(null);
    const [activeTab, setActiveTab] = useState<'dashboard' | 'reviews' | 'aspects'>('dashboard');

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    // Use environment variable for production, fallback to localhost for development
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://emotion-score-app.onrender.com";

    const handleUpload = async () => {
        if (!file) return;

        setAnalyzing(true);
        setProgress(10); // Start progress

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Fake progress for UX
            const interval = setInterval(() => {
                setProgress(prev => Math.min(prev + 5, 90));
            }, 500);

            const response = await axios.post(`${API_URL}/upload-dataset`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            clearInterval(interval);
            setProgress(100);

            // Fetch full results
            // In a real app we might fetch these separately or just use the response if modified to return all
            // For now let's assume the response has summary and we fetch others, or we just rely on the stored state in backend

            const summaryRes = await axios.get(`${API_URL}/analysis-summary`);
            const reviewsRes = await axios.get(`${API_URL}/reviews`);
            const aspectsRes = await axios.get(`${API_URL}/aspect-report`);

            setResults({
                summary: summaryRes.data,
                reviews: reviewsRes.data,
                aspects: aspectsRes.data
            });

        } catch (error) {
            console.error("Analysis failed", error);
            alert("Analysis failed. Please try again.");
            setProgress(0);
        } finally {
            setAnalyzing(false);
        }
    };

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

    if (!results) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-8 p-8">
                <div className="text-center space-y-4 max-w-2xl">
                    <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        Fake Review Detection System
                    </h1>
                    <p className="text-muted-foreground text-lg">
                        Upload your dataset (CSV) to detect fake reviews, analyze emotions, and uncover opinion manipulation using advanced AI.
                    </p>
                </div>

                <Card className="w-full max-w-md border-2 border-dashed border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Upload className="w-5 h-5 text-blue-500" />
                            Upload Dataset
                        </CardTitle>
                        <CardDescription>Supported format: CSV (must contain 'Text' column)</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <Input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="cursor-pointer bg-white dark:bg-slate-950"
                        />
                        {analyzing && (
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs text-muted-foreground">
                                    <span>Analyzing reviews...</span>
                                    <span>{progress}%</span>
                                </div>
                                <Progress value={progress} className="h-2" />
                            </div>
                        )}
                    </CardContent>
                    <CardFooter>
                        <Button
                            onClick={handleUpload}
                            disabled={!file || analyzing}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                        >
                            {analyzing ? "Processing..." : "Start Analysis"}
                        </Button>
                    </CardFooter>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl text-center">
                    {[
                        { icon: AlertTriangle, title: "Fake Detection", desc: "Autoencoder-based anomaly detection" },
                        { icon: Activity, title: "Emotion Analysis", desc: "RoBERTa-driven psychological profiling" },
                        { icon: Search, title: "Opinion Mining", desc: "Aspect-based sentiment extraction" }
                    ].map((feature, i) => (
                        <div key={i} className="flex flex-col items-center space-y-2 p-4 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                            <feature.icon className="w-8 h-8 text-slate-500" />
                            <h3 className="font-semibold">{feature.title}</h3>
                            <p className="text-sm text-muted-foreground">{feature.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Helper to render dashboard
    const renderDashboard = () => (
        <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                    { title: "Total Reviews", value: results.summary.total_reviews, icon: FileText, color: "text-slate-500" },
                    { title: "Fake Reviews", value: results.summary.fake_reviews, icon: AlertTriangle, color: "text-red-500" },
                    { title: "Genuine Reviews", value: results.summary.genuine_reviews, icon: CheckCircle, color: "text-green-500" },
                    { title: "Avg Trust Score", value: `${results.summary.avg_trust_score.toFixed(1)}%`, icon: Activity, color: "text-blue-500" },
                ].map((stat, i) => (
                    <Card key={i}>
                        <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
                            <stat.icon className={`h-4 w-4 ${stat.color}`} />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{stat.value}</div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Fake vs Genuine Chart */}
                <Card className="col-span-1">
                    <CardHeader>
                        <CardTitle>Distribution</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={[
                                        { name: "Fake", value: results.summary.fake_reviews },
                                        { name: "Genuine", value: results.summary.genuine_reviews }
                                    ]}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    <Cell fill="#ef4444" />
                                    <Cell fill="#22c55e" />
                                </Pie>
                                <Tooltip />
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Emotion Heatmap or Aspect Overview */}
                <Card className="col-span-1">
                    <CardHeader>
                        <CardTitle>Aspect Sentiment Analysis</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <RechartsBarChart
                                layout="vertical"
                                data={Object.entries(results.aspects).map(([key, val]) => ({
                                    name: key,
                                    score: val.avg_weighted_score,
                                    confidence: val.confidence_score
                                })).slice(0, 8)}
                                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                            >
                                <XAxis type="number" domain={[-1, 1]} />
                                <YAxis type="category" dataKey="name" width={100} />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="score" fill="#8884d8" name="Sentiment Score" />
                            </RechartsBarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>

            <div className="grid grid-cols-1 gap-6">
                <h2 className="text-2xl font-bold">Aspect Analysis & Summaries</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(results.aspects).map(([aspect, data]) => (
                        <Card key={aspect} className="flex flex-col">
                            <CardHeader className="pb-2">
                                <div className="flex justify-between items-center">
                                    <CardTitle className="capitalize">{aspect}</CardTitle>
                                    <Badge variant={data.avg_weighted_score > 0 ? "default" : "destructive"}>
                                        {data.dominant_sentiment}
                                    </Badge>
                                </div>
                            </CardHeader>
                            <CardContent className="flex-1 space-y-4">
                                <div className="text-sm text-muted-foreground p-3 bg-slate-50 dark:bg-slate-900 rounded-md">
                                    {data.summary || "No summary available."}
                                </div>
                                <div className="space-y-1">
                                    <div className="flex justify-between text-xs">
                                        <span>Confidence</span>
                                        <span>{data.confidence_score.toFixed(1)}%</span>
                                    </div>
                                    <Progress value={data.confidence_score} className="h-1.5" />
                                </div>
                                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                    <Activity className="w-3 h-3" />
                                    Dominant Emotion: <span className="font-semibold text-foreground">{data.dominant_emotion}</span>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>
        </div>
    );

    const renderReviews = () => (
        <div className="space-y-4">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">Detailed Review Analysis</h2>
                <span className="text-sm text-muted-foreground">Showing suspicious reviews first</span>
            </div>
            <div className="space-y-4">
                {results.reviews
                    .sort((a, b) => a.trust_score - b.trust_score) // Show lowest trust first
                    .slice(0, 50) // Limit to 50 for performance
                    .map((review) => (
                        <Card key={review.id} className={review.review_label === "Fake" ? "border-red-200 dark:border-red-900/50 bg-red-50/10" : ""}>
                            <CardHeader className="pb-2">
                                <div className="flex justify-between items-start">
                                    <div className="space-y-1">
                                        <div className="flex items-center gap-2">
                                            {review.review_label === "Fake" ?
                                                <Badge variant="destructive">Fake Detected</Badge> :
                                                <Badge variant="default" className="bg-green-600 hover:bg-green-700">Genuine</Badge>
                                            }
                                            <span className="text-xs text-muted-foreground">Trust Score: {review.trust_score.toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                <p className="text-sm">{review.Text}</p>
                                {review.review_label === "Fake" && (
                                    <div className="flex items-start gap-2 p-3 bg-red-100/50 dark:bg-red-900/20 rounded text-sm text-red-700 dark:text-red-300">
                                        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                                        <span><strong>Why it's fake:</strong> {review.fake_reason}</span>
                                    </div>
                                )}
                                <div className="grid grid-cols-2 gap-4 text-xs mt-2">
                                    {review.emotional_exaggeration === 1 && (
                                        <div className="text-orange-600 flex items-center gap-1">
                                            <Activity className="w-3 h-3" /> Emotional Exaggeration Detected
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    ))
                }
            </div>
        </div>
    )

    return (
        <div className="container mx-auto p-4 space-y-8 max-w-7xl">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold tracking-tight">Analysis Report</h1>
                <Button variant="outline" onClick={() => setResults(null)}>New Analysis</Button>
            </div>

            <div className="flex space-x-1 border-b">
                <Button
                    variant={activeTab === 'dashboard' ? 'default' : 'ghost'}
                    onClick={() => setActiveTab('dashboard')}
                    className="rounded-b-none"
                >
                    Dashboard
                </Button>
                <Button
                    variant={activeTab === 'reviews' ? 'default' : 'ghost'}
                    onClick={() => setActiveTab('reviews')}
                    className="rounded-b-none"
                >
                    Detailed Reviews
                </Button>
            </div>

            {activeTab === 'dashboard' && renderDashboard()}
            {activeTab === 'reviews' && renderReviews()}
        </div>
    );
}
