"use client";

import React, { useEffect, useRef } from "react";

interface Bubble {
    x: number;
    y: number;
    vx: number;
    vy: number;
    radius: number;
    color: string;
}

const BackgroundBubbles: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        let animationFrameId: number;
        let bubbles: Bubble[] = [];

        // Configuration
        const bubbleCount = 15;
        // Using HSLA for easier gradient manipulation if needed, but strings work fine
        const colors = [
            "rgba(6, 182, 212, 0.6)", // Cyan
            "rgba(139, 92, 246, 0.6)", // Violet
            "rgba(59, 130, 246, 0.6)", // Blue
        ];

        const init = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            bubbles = [];

            for (let i = 0; i < bubbleCount; i++) {
                const radius = Math.random() * 80 + 40; // Larger soft spheres
                bubbles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.8, // Slow, gentle drift
                    vy: (Math.random() - 0.5) * 0.8,
                    radius: radius,
                    color: colors[Math.floor(Math.random() * colors.length)],
                });
            }
        };

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas

            // Global Composite Operation for "glowing" intersection effect
            ctx.globalCompositeOperation = "screen";

            bubbles.forEach((bubble, index) => {
                // Update position
                bubble.x += bubble.vx;
                bubble.y += bubble.vy;

                // Wall Bounce
                if (bubble.x - bubble.radius < 0 || bubble.x + bubble.radius > canvas.width) bubble.vx *= -1;
                if (bubble.y - bubble.radius < 0 || bubble.y + bubble.radius > canvas.height) bubble.vy *= -1;

                // Draw Bubble
                const gradient = ctx.createRadialGradient(
                    bubble.x, bubble.y, 0,
                    bubble.x, bubble.y, bubble.radius
                );
                gradient.addColorStop(0, bubble.color); // Center color
                gradient.addColorStop(0.7, bubble.color.replace("0.6)", "0.1)")); // Soft fade
                gradient.addColorStop(1, "transparent"); // Edge

                ctx.beginPath();
                ctx.fillStyle = gradient;
                ctx.arc(bubble.x, bubble.y, bubble.radius, 0, Math.PI * 2);
                ctx.fill();

                // Simple approach to avoid complex collision logic bugs:
                // Just let them pass through with the screen blend mode which looks like 'light' mixing
            });

            ctx.globalCompositeOperation = "source-over"; // Reset
            animationFrameId = requestAnimationFrame(draw);
        };

        // Handle resize
        const handleResize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            init(); // Re-init bubbles to fit screen
        };

        window.addEventListener("resize", handleResize);
        init();
        draw();

        return () => {
            window.removeEventListener("resize", handleResize);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none z-0"
            style={{ filter: "blur(20px)", opacity: 0.8 }} // Heavy blur for "glowing orb" effect
        />
    );
};

export default BackgroundBubbles;
