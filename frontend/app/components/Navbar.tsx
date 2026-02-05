"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { JSX } from "react";

export default function Navbar(): JSX.Element {
  const pathname = usePathname();

  const linkClass = (href: string): string =>
    `px-4 py-2 rounded transition ${
      pathname === href
        ? "bg-blue-600 text-white"
        : "text-gray-700 hover:bg-gray-100"
    }`;

  return (
    <nav className="bg-white shadow sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex justify-between items-center">
        <Link href="/" className="text-xl font-bold text-blue-600">
          ❤️ HeartPredict AI
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
      </div>
    </nav>
  );
}
