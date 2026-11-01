import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AIRA — Autonomous Intelligent Research Assistant",
  description: "Agentic research orchestration for literature review, comparison, and synthesis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans bg-slate-950 text-slate-100">{children}</body>
    </html>
  );
}
