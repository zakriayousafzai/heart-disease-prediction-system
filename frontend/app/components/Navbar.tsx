"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { JSX } from "react";

export default function Navbar(): JSX.Element {
  const pathname = usePathname();

  const linkClass = (href: string): string =>
    `px-4 py-2 rounded transition ${
      pathname === href
        ? "bg-[#f7f4f3] text-[#5b2333]"
        : "text-[#f7f4f3] hover:shadow-[inset_0px_0px_4px_rgba(0,0,0,0.6)] hover:bg-[#5b2333]"
    }`;

  return (
    <nav className="bg-[#5b2333] shadow sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex justify-between items-center">
        <Link href="/" className="text-xl font-bold text-[#f7f4f3]">
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
        <div className="bg-[#f7f4f3] h-10 w-10 rounded-full flex items-center justify-center">
           
        </div>
      </div>
    </nav>
  );
}
