import { useQuery, useQueryClient } from "@tanstack/react-query";
import { StyledWrapper } from "./styled";
import { Typography, Stack, Divider } from "@mui/material";
import {
  getYachts,
  getNewArrivals,
  getPersonalizedNewArrivals,
  getRecommendedYachts,
  getTopBookedYachts,
} from "src/services/yachts";
import { YachtCard, Loader } from "src/components";
import { useAuth } from "src/context/authContext";

const HomePage = () => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  // const {
  //   isPending,
  //   error,
  //   data: yachts,
  // } = useQuery({
  //   queryKey: ["yachts"],
  //   queryFn: getYachts,
  // });

  const { data: recommendedYachts } = useQuery({
    queryKey: ["recommendedYachts", user && user.id],
    queryFn: getRecommendedYachts,
    enabled: !!user?.id,
  });

  const { data: topBookedYachts, isPending } = useQuery({
    queryKey: ["topBooked"],
    queryFn: getTopBookedYachts,
    enabled: !user?.id,
  });

  const { data: newArrivals, isNewArrivalsPending } = useQuery({
    queryKey: ["newArrivals"],
    queryFn: getNewArrivals,
    enabled: !user?.id,
  });

  const { data: personalizedNewArrivals, isPersonalizedNewArrivalsPending } =
    useQuery({
      queryKey: ["personalizedNewArrivals"],
      queryFn: getPersonalizedNewArrivals,
      enabled: !!user?.id,
    });

  topBookedYachts?.forEach((yacht) => {
    queryClient.setQueryData(["yachts", yacht.id], yacht);
  });

  newArrivals?.forEach((yacht) => {
    queryClient.setQueryData(["yachts", yacht.id], yacht);
  });

  personalizedNewArrivals?.forEach((yacht) => {
    queryClient.setQueryData(["yachts", yacht.id], yacht);
  });

  recommendedYachts?.forEach((yacht) => {
    queryClient.setQueryData(["yachts", yacht.id], yacht);
  });

  // set filters to search params
  return (
    <>
      <StyledWrapper>
        <Stack width={"60%"}>
          <Typography variant="subtitle" color={"special.white"}>
            From sailing dreams to unforgettable escapes
          </Typography>
          <Typography variant="h1" color={"special.white"}>
            Choose the Perfect Yacht for Your Voyage
          </Typography>
        </Stack>
      </StyledWrapper>

      {user?.id ? (
        <>
          <Stack gap={4} padding={4}>
            <Typography variant="h5">Top recommended yachts for you</Typography>
            <Stack
              direction={"row"}
              gap={4}
              rowGap={4}
              flexWrap={"wrap"}
              justifyContent={"space-between"}
            >
              {recommendedYachts?.map((yacht) => (
                <YachtCard key={yacht.id} yachtDetails={yacht} />
              ))}
            </Stack>

            <Divider color="secondary.contrastText" />
            <Typography variant="h5">New arrivals for you</Typography>
            <Stack
              direction={"row"}
              gap={4}
              rowGap={4}
              flexWrap={"wrap"}
              justifyContent={"space-between"}
            >
              {isPersonalizedNewArrivalsPending && <Loader />}
              {personalizedNewArrivals?.map((yacht) => (
                <YachtCard key={yacht.id} yachtDetails={yacht} />
              ))}
            </Stack>
          </Stack>
        </>
      ) : (
        <>
          <Stack gap={4} padding={4}>
            <Typography variant="h5">Top booked yachts</Typography>
            <Stack
              direction={"row"}
              gap={4}
              rowGap={4}
              flexWrap={"wrap"}
              justifyContent={"space-between"}
            >
              {isPending && <Loader />}
              {topBookedYachts?.map((yacht) => (
                <YachtCard key={yacht.id} yachtDetails={yacht} />
              ))}
            </Stack>

            <Divider color="secondary.contrastText" />
            <Typography variant="h5">New arrivals</Typography>
            <Stack
              direction={"row"}
              gap={4}
              rowGap={4}
              flexWrap={"wrap"}
              justifyContent={"space-between"}
            >
              {isNewArrivalsPending && <Loader />}
              {newArrivals?.map((yacht) => (
                <YachtCard key={yacht.id} yachtDetails={yacht} />
              ))}
            </Stack>
          </Stack>
        </>
      )}
    </>
  );
};

export default HomePage;
