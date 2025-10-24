import { use, useEffect, useRef } from "react";

export const useIntersectionObserver = (
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage
) => {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !hasNextPage) return;

    const options = {
      root: null,
      rootMargin: "100px",
      threshold: 0,
    };

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !isFetchingNextPage) {
        fetchNextPage();
      }
    }, options);

    observer.observe(ref.current);

    return () => observer.disconnect();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage, ref]);

  return ref;
};
