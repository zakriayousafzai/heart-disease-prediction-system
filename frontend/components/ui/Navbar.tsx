"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { JSX, useState } from "react";
import "./Navbar.css";
import Image from "next/image";

export default function Navbar(): JSX.Element {
  const pathname = usePathname();

  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const linkClass = (href: string): string =>
    `px-4 py-1 rounded-lg transition ${
      pathname === href
        ? "bg-brand-fg text-brand-bg"
        : "text-brand-fg hover:shadow-[inset_0px_0px_4px_rgba(0,0,0,0.6)] hover:bg-brand-bg"
    }`;

  return (
    <nav className="bg-black/15 backdrop-blur-md border border-white/20 rounded-full m-5 shadow sticky top-5 z-50">
      <div className="max-w-7xl mx-auto px-7 h-16 flex justify-between items-center">
        <Link href="/" className="text-xl font-bold text-brand-fg">
          HeartPredict AI
        </Link>

        <div className="flex gap-2 max-[630px]:hidden">
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
        <div className="group bg-brand-fg h-10 w-10 rounded-full flex items-center justify-center">
          <Image className="max-[630px]:hidden group-hover-heartbeat" src={"./heart.svg"} height={30} width={30} alt="heart"/>
          <div className="min-[630px]:hidden">

          <div className="hamburger-icon" onClick={toggleMenu}>
            <div className="bar"></div>
            <div className="bar"></div>
            <div className="bar"></div>
          </div>

          <ul className={`nav-links ${isOpen ? "open" : ""} rounded-2xl bg-black/50`}>
            <li>
              <Link href="/predict" className={linkClass("/predict")}>
                Predict
              </Link>
            </li>
            <li>
              <Link href="/history" className={linkClass("/history")}>
                History
              </Link>
            </li>
            <li>
              <Link href="/metrics" className={linkClass("/metrics")}>
                Metrics
              </Link>
            </li>
          </ul>
          </div>
        </div>
      </div>
    </nav>
  );
}
