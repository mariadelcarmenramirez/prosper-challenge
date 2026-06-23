okay let's move to the agentic workflow behaviour and start coding. Every tool maps directly to one fastAPI endpoint. According to this:

Build a simple EHR service (any framework you like) that exposes at least these endpoints:

create_patient — register a new patient (e.g. name, date of birth, contact info)
find_patient — look up an existing patient by name and date of birth
list_availability_slots — return the clinic's available appointment slots for a given date or range
create_appointment — book a slot for a given patient
cancel_appointment — cancel an existing appointment
The EHR should persist its data in a database so patients and appointments survive across restarts — please don't keep state in memory only. The shape of the request/response is up to you — design it the way you'd want a real integration to look.

We are asuming each appointment last an hour. 
We identify patients by patients_id but a patient object also has (name, date of birth and phone number)
We identify appointments by appointment_id but an appointment has also associated a patient_id, a day and an hour.
The two objects are patients and appointments, i will identify them with id, first (patient_id) to make it easier their identification and second (appointment_id), in case in the future we have more than one doctor. Right now i will start assuming only one doctor (which means that if an hour has an appointment this hour cannot be used to book a new appointment).
Clinic has clinic working hours which are MONDAY TO FRIDAY FROM 9AM TO 6PM. New appointments must be booked in clinic_hours 
I think the agent has to know the exact hour of today for every query, so it knows what that things like "this friday" means.

If something of this does not make sense to you feel free to tell me.

My agent will behave in the following way:

1. Agent receives the initial test after it has been processed by the STT service. Agent has to behave accordingly to identify the caller. I guess ask for name, date of birth and phone number. With this information checks inthe Database if the patient is there, using find_patient(name, DOB, phone_number).

2. Agents gets an output, information patient object (including patient_id) or null depending if patient exists or not. 
i. if pateint exists ask the user if they need a new appointment or to cancel an existing appointment. 
    a. if patient wants to cancel an appointment. Look towards a list of patient appointments (2. "List patient appointments" should probably filter to future + scheduled only.
When cancelling, listing all appointments would read back past and already-cancelled ones. The agent should only offer cancellable appointments — i.e. status = scheduled AND starts_at > now. Small thing, easy to miss.)(by Add a small lookup (e.g. GET /appointments?patient_id=...&day=...&hour=...), or a "list this patient's appointments" endpoint and let the agent pick), and list all the time(s), ask the patient for which of the appointments want to cancell listing them in case there is more than one or just saying "you have one on July 3rd at 10am, cancel that?". 
    ```
    when deleting an appointment i want to do soft delate:

    Soft delete (recommended)
    The row stays, but you flip a status column from scheduled → cancelled. The appointment still exists in the DB, just marked as cancelled.

    Keeps history / audit trail (important in a real medical context)
    This is why you have a status enum in the first place
    ```
    b. if patient wants to book a new appointment, ask patient for date (month and day) they are available , range of days, or range of hours, this is a little bit difficult to handle all the posibilities. Think a good way to do it, to handle date, range, maybe a list of available hours saved as patient_availability=[HOUR-DAY-MONTH, ...]. We are assuming we are in 2026. Include in the system prompt dynamically what day is today****, to translate things like this wednesday to the exact date **. 

    NEXT STEP: ONCE agents has receive from the user their availability, look in DB for existing appointment, and delete all the hours from the ones that are in patient_availability that are already booked or are outside of clinic_hours. 
    
    If once you have check the available hours, if the reduced list is EMPTY, which means that there are not available hours for appointment in the patient_availability list, make the agent ask the user for availability after the last HOUR-DAY-MONTH-YEAR, then patient is supposed to give an updated patient_availability AND you make a loop going back to "NEXT STEP: ONCE agents has receive from the user their availability, look in DB for existing appointment, and delete all the hours...". Maybe repeat these steps 4 times with the EMPTY flow, no more, the last time just say sorry we do not have availability right now, in a very polite way and finish the conversation.
    
    If the reduced list of hours is NOT EMPTY, tell the user that you will book the first one available which is HOUR-DAY-MONTH***. wait for confirmation, 
        * if user confirms, create the appointment which is register the appointment in the database with the app id, patient id and hour,day,month,year. Save it in the DB for the future. 
        * if user rejects the proposed slot, tell him the next one until the list is empty (another loop here), once the list is empty go to "If once you have check the available hours, if the reduced list is EMPTY, ...". Somehow try to count also the number of times the user has rejected the proposed slot -> ask for new availability list  -> reject the whole reduced list. So I avoid another infinite loop here (less probable but plausible). Maybe repeat this whole flow 4 times, no more, the last time just say sorry we do not have availability right now, in a very polite way and finish the conversation.
    


    **In both, a and b every time the user says something like I'm available/I want to cancel the appointment of Monday at 3pmm, the agent has to ask the user something like Monday XXth of MONTH at 3pm? To avoid ambiguity.

    *** Every time the agent proposes HOUR-DAY-MONTH-YEAR for appointment, Reserve the date with a status + expiry: slot becomes held****** with a timestamp, released after N minutes. In order to save the date in the database, so it cannot be saved by someone else.

    **** 
    ```
from datetime import datetime
from zoneinfo import ZoneInfo

def build_system_prompt() -> str:
    now = datetime.now(ZoneInfo("Europe/Madrid"))
    today_str = now.strftime("%A, %Y-%m-%d")  # e.g. "Tuesday, 2026-06-23"

    return f"""You are a friendly medical receptionist for a clinic.

Today is {today_str}. Use this to resolve relative dates like
"this Wednesday" or "next Monday" into exact calendar dates.
    ```

    *****
    REMARKS:
    - Past dates: the API should reject booking a slot in the past, even if the LLM resolves a relative date wrong. Your normalize-on-API-side note covers this — just don't forget to actually implement the check.
    - Timezone consistency: you're using Europe/Madrid in the prompt — store appointment times in the same zone (or UTC consistently) so "10am" means the same thing in the DB and the conversation.

    ******
    - A held slot must be excluded from list_availability_slots — not just scheduled ones. So your availability query now filters status NOT IN (scheduled, held WHERE not expired).
         Your list_availability_slots now has to exclude a slot if it has either a scheduled appointment or a non-expired held one — and include it if the only thing on it is a cancelled appointment. Just make sure that one query gets all three states right. 
    - Expiry needs a trigger. A held_until timestamp doesn't release itself. You need to treat a hold as expired at query time (held AND held_until > now() counts as taken; otherwise it's free).
    - Confirmation must convert hold → scheduled, and that conversion should re-check the hold hasn't expired in the meantime.

ii. if patient does not exists,    - if NOT FOUND → "I don't have you in our system, let me register you"
                  → call create_patient(name, dob, phone)  ← you already collected these, and create the new patient_id in an unique, DB generate the UUID.
                  → then proceed to "new appointment or cancel?" which is the i.a. and i.b flows.