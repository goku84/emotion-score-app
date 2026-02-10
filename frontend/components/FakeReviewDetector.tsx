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
import BackgroundBubbles from './BackgroundBubbles';

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
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
            <div className="relative flex flex-col items-center justify-center min-h-screen w-full overflow-hidden bg-slate-950 text-slate-100 selection:bg-cyan-500/30">
                {/* --- Ambient Background Animations --- */}
                {/* --- Ambient Background Animations --- */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    {/* Grid Pattern - Kept very subtle texture */}
                    <div className="absolute inset-0 bg-grid-pattern opacity-[0.02]" />

                    {/* New Physics Bubbles Component */}
                    <BackgroundBubbles />

                    {/* Static decorative data lines - kept for depth */}
                    <div className="absolute inset-0 bg-[radial-gradient(circle_800px_at_50%_50%,rgba(6,182,212,0.02),transparent)]" />
                </div>

                {/* --- Content Container --- */}
                <div className="relative z-10 flex flex-col items-center space-y-10 max-w-2xl w-full px-4">

                    {/* Header */}
                    <div className="text-center space-y-2 animate-[slide-up_0.7s_ease-out]">
                        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white drop-shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                            Dataset Upload
                        </h1>
                        <p className="text-cyan-400 font-medium tracking-[0.2em] text-sm uppercase">
                            AI / ML Readiness Interface
                        </p>
                    </div>

                    {/* Upload Zone */}
                    <div className="relative group">
                        {/* The Circular Drop Zone */}
                        <div className={`
                            relative z-0 flex flex-col items-center justify-center
                            w-72 h-72 md:w-80 md:h-80 rounded-full
                            border-2 border-dashed transition-all duration-500 ease-out
                            ${file
                                ? "border-cyan-400 bg-cyan-950/20 shadow-[0_0_30px_rgba(6,182,212,0.15)]"
                                : "border-slate-700 hover:border-cyan-500/50 hover:bg-slate-900/50"
                            }
                            ${analyzing ? "scale-95 opacity-50 cursor-not-allowed" : "scale-100"}
                        `}>

                            {/* Animated ring when active */}
                            <div className={`absolute inset-0 rounded-full border border-cyan-500/30 scale-110 opacity-0 transition-opacity duration-500 ${file ? "opacity-100 animate-pulse" : "group-hover:opacity-100"}`} />

                            {/* Actual File Input (Hidden Overlay) */}
                            {!analyzing && (
                                <input
                                    type="file"
                                    accept=".csv"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-50"
                                    title="Drop dataset here"
                                />
                            )}

                            {/* State: Analyzing (Full Circle Animation) */}
                            {analyzing ? (
                                <div className="absolute inset-0 z-50 flex items-center justify-center rounded-full overflow-hidden bg-slate-950/80 backdrop-blur-sm animate-[zoom-in_0.3s_ease-out]">
                                    {/* Rotating Radar Effect */}
                                    <div className="absolute inset-0 rounded-full radar-loader" />

                                    {/* Content inside loader */}
                                    <div className="relative z-10 flex flex-col items-center justify-center text-center space-y-2">
                                        <div className="text-4xl font-bold font-mono text-cyan-400 drop-shadow-[0_0_10px_rgba(6,182,212,0.5)]">
                                            {progress}%
                                        </div>
                                        <div className="text-xs text-cyan-300/80 tracking-widest uppercase animate-pulse">
                                            Analyzing Dataset
                                        </div>
                                        {/* Optional scanning line effect */}
                                        <div className="absolute inset-0 w-full h-[2px] bg-cyan-400/50 shadow-[0_0_15px_cyan] animate-[scan-line_2s_linear_infinite]" />
                                    </div>
                                </div>
                            ) : file ? (
                                // State: File Selected
                                <div className="flex flex-col items-center space-y-4 animate-[zoom-in_0.3s_ease-out] z-10 pointer-events-none">
                                    <div className="w-16 h-16 rounded-full bg-cyan-400/20 flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.2)]">
                                        <FileText className="w-8 h-8 text-cyan-400" />
                                    </div>
                                    <div className="text-center">
                                        <p className="text-white font-medium truncate max-w-[200px]">{file.name}</p>
                                        <p className="text-xs text-cyan-400/70 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
                                    </div>
                                    <p className="text-slate-400 text-xs mt-2">Click to change file</p>
                                </div>
                            ) : (
                                // State: Default (No File)
                                <div className="flex flex-col items-center space-y-4 text-center p-6 z-10 pointer-events-none">
                                    <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center group-hover:bg-cyan-900/20 transition-colors duration-300">
                                        <Upload className="w-8 h-8 text-slate-400 group-hover:text-cyan-400 transition-colors duration-300" />
                                    </div>
                                    <div className="space-y-1">
                                        <p className="text-lg font-medium text-slate-200 group-hover:text-white transition-colors">Drag dataset here</p>
                                        <p className="text-sm text-slate-500 uppercase tracking-wide group-hover:text-cyan-400/70 transition-colors">or browse</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Action Buttons (External to circle to avoid click conflicts if needed, or overlay) */}
                        {file && !analyzing && (
                            <div className="absolute -bottom-20 left-1/2 -translate-x-1/2 w-48 animate-[slide-up_0.5s_ease-out]">
                                <Button
                                    onClick={handleUpload}
                                    className="w-full bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-900/20 border border-cyan-400/20 rounded-full h-12 text-md font-medium tracking-wide"
                                >
                                    START ANALYSIS
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* Footer Info */}
                    {!file && !analyzing && (
                        <div className="grid grid-cols-3 gap-8 pt-8 text-center opacity-60 hover:opacity-100 transition-opacity duration-500">
                            {[
                                { title: "CSV SUPPORT", desc: "Structured Data" },
                                { title: "AUTO-DETECT", desc: "Fake Patterns" },
                                { title: "SECURE", desc: "Local Processing" } // or similar
                            ].map((item, i) => (
                                <div key={i} className="flex flex-col items-center">
                                    <p className="text-xs font-bold text-slate-300 tracking-wider mb-1">{item.title}</p>
                                    <p className="text-[10px] text-cyan-500/70 uppercase">{item.desc}</p>
                                </div>
                            ))}
                        </div>
                    )}
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
