"use client";

import React from "react";

/**
 * Lightweight Markdown renderer for backend-generated report sections.
 *
 * The report service emits a small Markdown subset (### / #### headings,
 * "- " bullets, numbered lists, pipe tables, **bold**, *italic*, `code`).
 * Rendering it as raw text produced a dense wall of symbols, so this turns
 * it into properly spaced elements. `relabel` lets the caller swap raw paper
 * IDs (DOIs) for readable labels before display.
 */

type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "table"; header: string[]; rows: string[][] };

const splitRow = (line: string): string[] =>
  line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());

const isTableSeparator = (line: string) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      blocks.push({ kind: "paragraph", text: para.join(" ") });
      para = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushPara();
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushPara();
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2] });
      continue;
    }

    if (trimmed.startsWith("|")) {
      flushPara();
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      i--;
      const rows = tableLines.filter((l) => !isTableSeparator(l)).map(splitRow);
      if (rows.length) blocks.push({ kind: "table", header: rows[0], rows: rows.slice(1) });
      continue;
    }

    const bullet = trimmed.match(/^[-*•]\s+(.*)$/);
    const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushPara();
      const ordered = !!numbered && !bullet;
      const text = (bullet || numbered)![1];
      const last = blocks[blocks.length - 1];
      if (last && last.kind === "list" && last.ordered === ordered) last.items.push(text);
      else blocks.push({ kind: "list", ordered, items: [text] });
      continue;
    }

    // Indented continuation of a list item.
    const last = blocks[blocks.length - 1];
    if (!para.length && last && last.kind === "list" && /^\s{2,}/.test(line)) {
      last.items[last.items.length - 1] += " " + trimmed;
      continue;
    }

    para.push(trimmed);
  }
  flushPara();
  return blocks;
}

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g);
  return parts.map((part, idx) => {
    const key = `${keyPrefix}-${idx}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={key} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={key} className="font-mono text-[0.9em] bg-slate-800/70 text-slate-200 px-1.5 py-0.5 rounded">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return (
        <em key={key} className="text-slate-400">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <React.Fragment key={key}>{part}</React.Fragment>;
  });
}

interface MarkdownTextProps {
  text: string;
  relabel?: (text: string) => string;
  className?: string;
}

export const MarkdownText: React.FC<MarkdownTextProps> = ({ text, relabel, className = "" }) => {
  const blocks = parseBlocks(relabel ? relabel(text) : text);

  return (
    <div className={`space-y-4 text-[15px] leading-7 text-slate-300 ${className}`}>
      {blocks.map((b, bIdx) => {
        const key = `b${bIdx}`;
        switch (b.kind) {
          case "heading":
            return b.level <= 3 ? (
              <h5 key={key} className="text-base font-semibold text-white pt-3 first:pt-0">
                {renderInline(b.text, key)}
              </h5>
            ) : (
              <h6 key={key} className="text-[13px] font-semibold uppercase tracking-wider text-slate-400 pt-2 first:pt-0">
                {renderInline(b.text, key)}
              </h6>
            );
          case "paragraph":
            return <p key={key}>{renderInline(b.text, key)}</p>;
          case "list": {
            const ListTag = b.ordered ? "ol" : "ul";
            return (
              <ListTag
                key={key}
                className={`space-y-3 pl-5 ${b.ordered ? "list-decimal" : "list-disc"} marker:text-slate-600`}
              >
                {b.items.map((item, iIdx) => (
                  <li key={`${key}-${iIdx}`} className="pl-1">
                    {renderInline(item, `${key}-${iIdx}`)}
                  </li>
                ))}
              </ListTag>
            );
          }
          case "table":
            return (
              <div key={key} className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-900 text-slate-400 text-xs uppercase tracking-wider">
                    <tr>
                      {b.header.map((h, hIdx) => (
                        <th key={hIdx} className="px-4 py-2.5 font-semibold whitespace-nowrap">
                          {renderInline(h, `${key}-h${hIdx}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80">
                    {b.rows.map((row, rIdx) => (
                      <tr key={rIdx} className="hover:bg-slate-900/50">
                        {row.map((cell, cIdx) => (
                          <td key={cIdx} className="px-4 py-2.5 text-slate-300 align-top">
                            {renderInline(cell, `${key}-${rIdx}-${cIdx}`)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
        }
      })}
    </div>
  );
};
