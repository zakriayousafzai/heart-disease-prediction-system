import { JSX } from "react";
import "./globals.css";
import Navbar from "@/app/components/Navbar";

export const metadata = {
  title: "Heart Disease Prediction System",
  description: "AI-based Heart Disease Risk Prediction",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): JSX.Element {
  return (
    <html lang="en">
      <body className="">
        <Navbar />
        <main className="max-w-7xl mx-auto p-4">{children}</main>
      </body>
    </html>
  );
}
