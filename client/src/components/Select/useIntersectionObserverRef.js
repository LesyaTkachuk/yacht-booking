"use client";

import { useCallback, useRef } from "react";

// this intersection observer causes more rerenders
// should be used only for cases when fetching items are mounted conditionally on the page (like in Select dropdown)
const useIntersectionObserverRef = (
  isFetchingNextPage,
  hasNextPage,
  fetchNextPage
) => {
  const observer = (useRef < IntersectionObserver) | (null > null);

  const lastEntryRef = useCallback(
    (node) => {
      if (isFetchingNextPage || !fetchNextPage) return;

      if (observer.current) observer.current.disconnect();

      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasNextPage) {
          fetchNextPage();
        }
      });

      if (node) observer.current.observe(node);
    },
    [isFetchingNextPage, hasNextPage, fetchNextPage]
  );

  return lastEntryRef;
};

export default useIntersectionObserverRef;
