"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { JSX } from "react";

export default function Navbar(): JSX.Element {
  const pathname = usePathname();

  const linkClass = (href: string): string =>
    `px-4 py-2 rounded transition ${
      pathname === href
        ? "bg-brand-fg text-brand-bg"
        : "text-brand-fg hover:shadow-[inset_0px_0px_4px_rgba(0,0,0,0.6)] hover:bg-brand-bg"
    }`;

  return (
    <nav className="bg-black/10 backdrop-blur-md border border-white/20 rounded-full m-5 shadow sticky top-5 z-50">
      <div className="max-w-7xl mx-auto px-7 h-16 flex justify-between items-center">
        <Link href="/" className="text-xl font-bold text-brand-fg">
          HeartPredict AI
        </Link>

        <div className="flex gap-2">
          <Link href="/predict" className={linkClass("/predict")}>
            Predict
          </Link>
          <Link href="/history" className={linkClass("/history")}>
            History
          </Link>
          <Link href="/metrics" className={linkClass("/metrics")}>
            Metrics
          </Link>
        </div>
        <div className="bg-brand-fg h-10 w-10 rounded-full flex items-center justify-center"/>
      </div>
    </nav>
  );
}
